# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import numpy as np
from tqdm import tqdm
from ultralytics.utils import LOGGER
from ultralytics.utils.loss3d import Loss3D
from ultralytics.data.augment3d import Augment3D
from copy import copy
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import strip_optimizer
from ultralytics.nn import yaml_model_load
from ultralytics.nn.tasks import DetectionModel3D

class Dataset3D(Dataset):
    """Custom Dataset for 3D YOLO training."""
    def __init__(self, data_dir, split='train', transform=None):
        """Initialize Dataset3D."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.volume_files = list(self.data_dir.glob('**/*.npy'))  # Adjust file pattern as needed
        
        if not self.volume_files:
            raise FileNotFoundError(f'No volume files found in {self.data_dir}')
        
        LOGGER.info(f'Found {len(self.volume_files)} volume files in {self.data_dir}')

    def __len__(self):
        """Return the total number of samples."""
        return len(self.volume_files)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        volume_path = self.volume_files[idx]
        
        # Load volume (implement your loading logic here)
        volume = self.load_volume(volume_path)
        label_path = Path(str(volume_path).replace("/volumes/", "/labels/"))
        labels = self.load_labels(label_path)
        
        # Apply transforms if any
        if self.transform:
            volume, labels = self.transform(volume, labels)
            
        return {
            'volume': torch.FloatTensor(volume),
            'labels': torch.FloatTensor(labels),
            'path': str(volume_path),
        }

    def load_volume(self, path):
        """Load a 3D volume."""
        # Implement your volume loading logic here
        # Example using nibabel:
        # import nibabel as nib
        volume = np.load(path)
        return volume

    def load_labels(self, volume_path):
        """Load labels for a volume."""
        # Implement your label loading logic here
        # Example: load from a corresponding label file
        label_path = volume_path.with_suffix('.txt')
        if label_path.exists():
            # Load and parse labels
            labels = []
            with open(label_path) as f:
                for line in f:
                    # Parse label format: class x y z width height depth
                    values = list(map(float, line.strip().split()))
                    labels.append(values)
            return labels
        return []

class BaseTrainer3D(BaseTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """Initialize a 3D detection trainer."""
        super().__init__(cfg, overrides, _callbacks)
        if overrides is None:
            overrides = {}
        self.overrides = overrides
        # self.args.update(overrides)
        
        if self.args.data:
            self.setup_data_paths()
        self.setup_augmentations()
        # Initialize data loaders as None
        self.train_loader = None
        self.valid_loader = None

    def setup_data_paths(self):
        """Setup data directory paths."""
        data_path = Path(self.args.data)
        
        # If data is a YAML file, load it
        if data_path.suffix in ('.yaml', '.yml'):
            import yaml
            try:
                with open(data_path) as f:
                    data_dict = yaml.safe_load(f)
                self.args.data_dir = Path(data_dict.get('path', './datasets'))
                self.args.train_dir = Path(data_dict.get('train', 'train'))
                self.args.val_dir = Path(data_dict.get('val', 'val'))
            except Exception as e:
                LOGGER.warning(f"Failed to load data YAML: {e}")
                self.args.data_dir = data_path.parent
        else:
            # Assume data is a directory
            self.args.data_dir = data_path
            self.args.train_dir = data_path / 'train'
            self.args.val_dir = data_path / 'val'
            
        # Verify directories exist
        if not self.args.data_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.args.data_dir}' does not exist")
        
        LOGGER.info(f"Data directory set to: {self.args.data_dir}")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO3D detection model."""
        if cfg:
            if isinstance(cfg, (str, Path)):
                cfg = yaml_model_load(cfg)
            self.model = DetectionModel3D(cfg=cfg, verbose=verbose and RANK == -1)
            self.model.to(self.device)
            return self.model
        
        if weights:
            # self.model = attempt_load_one_weight(weights)
            return self.model
            
        raise NotImplementedError("Either cfg or weights must be provided to get_model()")

    def setup_augmentations(self):
        """Setup data augmentation pipeline."""
        self.augment = Augment3D(
            size=self.args.imgsz,
            scale=(0.8, 1.2),
            translate=0.2,
            flip_prob=0.5,
            rotate=15
        )
        
        # You can create more complex augmentation pipelines using Compose3D
        # self.augment = Compose3D([
        #     Augment3D(size=self.args.imgsz),
        #     # Add more augmentations here
        # ])
        
        LOGGER.info("Initialized 3D data augmentation pipeline")

    def setup_train_loader(self):
        """Create and setup training data loader."""
        LOGGER.info(f"{colorstr('train:')} Scanning training data...")
        
        # Create dataset
        train_dataset = Dataset3D(
            data_dir=self.args.data_dir / 'volumes/train',
            split='train',
            transform=self.augment  # Your 3D augmentation class
        )
        
        # Create data loader
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        LOGGER.info(f"{colorstr('train:')} Created DataLoader with {len(train_dataset)} samples")

    def setup_valid_loader(self):
        """Create and setup validation data loader."""
        LOGGER.info(f"{colorstr('val:')} Scanning validation data...")
        
        # Create dataset
        valid_dataset = Dataset3D(
            data_dir=self.args.data_dir / 'volumes/val',
            split='val',
            transform=None  # Usually no augmentation for validation
        )
        
        # Create data loader
        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.args.batch,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        LOGGER.info(f"{colorstr('val:')} Created DataLoader with {len(valid_dataset)} samples")

    def train(self):
        """Start training process."""
        # Setup data loaders if not already set
        if self.train_loader is None:
            self.setup_train_loader()
        if self.valid_loader is None:
            self.setup_valid_loader()
            
        # Continue with training
        return super().train()

    @staticmethod
    def collate_fn(batch):
        """Collate data for the data loader."""
        volumes = torch.stack([item['volume'] for item in batch])
        labels = [item['labels'] for item in batch]
        paths = [item['path'] for item in batch]
        
        return {
            'volume': volumes,
            'labels': labels,
            'paths': paths,
        }

    def preprocess_batch(self, batch):
        """Preprocess a batch of 3D data."""
        # Move volume to device
        batch["volume"] = batch["volume"].to(self.device, non_blocking=True)
        
        # Move each label tensor to device while keeping as list
        batch["labels"] = [
            torch.tensor(labels, device=self.device) 
            if len(labels) > 0 
            else torch.zeros((0, 7), device=self.device)
            for labels in batch["labels"]
        ]
        
        return batch

    def progress_string(self):
        """Return a formatted string showing training progress."""
        return super().progress_string()

    def update_model(self):
        """Update the model parameters."""
        # Add any 3D-specific model updates here
        super().update_model()

    def get_validator(self):
        """Return an instance of the 3D detection validator."""
        self.loss_names = ['box_loss']  # reset loss names for 3D detection
        return super().get_validator()

    def save_model(self):
        """Save model checkpoints."""
        super().save_model()

    def train_step(self, batch):
        """Training step."""
        # Forward
        preds = self.model(batch["volume"])
        
        # Calculate loss
        loss, loss_items = self.loss_fn(preds, batch["labels"])
        
        # Backward
        loss.backward()
        
        # Optimize
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss_items
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.args.epochs}")
        
        for batch in pbar:
            batch = self.preprocess_batch(batch)
            
            # Apply augmentation
            batch["volume"], batch["labels"] = self.augment(batch["volume"], batch["labels"])
            
            # Train step
            loss_items = self.train_step(batch)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_items[0]:.4f}"})
            
        return loss_items
    
    def train(self):
        if self.train_loader is None:
            self.setup_train_loader()
        if self.valid_loader is None:
            self.setup_valid_loader()
        """Training loop."""
        for self.epoch in range(self.args.epochs):
            # Train epoch
            loss_items = self.train_epoch()
            
            # Validation
            if self.epoch % self.args.val_interval == 0:
                stats = self.validator.evaluate()
                
            # Save checkpoint
            if self.epoch % self.args.save_interval == 0:
                self.save_model()
            
            # Update scheduler
            self.scheduler.step()
            
        LOGGER.info("Training completed.")

class DetectionTrainer3D(BaseTrainer3D):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """Initialize a 3D detection trainer with specific settings."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'detect3d'

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model for training."""
        model = super().get_model(cfg, weights, verbose)
        self.args.model = model
        return model

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build a 3D detection dataset."""
        # Implement 3D dataset building logic here
        return super().build_dataset(img_path, mode, batch)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Return a 3D detection dataloader."""
        # Implement 3D dataloader logic here
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def preprocess_batch(self, batch):
        """Preprocess batch of 3D data."""
        # Add 3D-specific preprocessing here
        return super().preprocess_batch(batch)

    def progress_string(self):
        """Return a string describing training progress."""
        return super().progress_string()

    def get_validator(self):
        """Return an instance of the 3D detection validator."""
        self.loss_names = ['box3d_loss']
        return super().get_validator() 