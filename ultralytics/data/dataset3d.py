# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import LOGGER, TQDM

class Dataset3D(Dataset):
    """Custom Dataset for 3D YOLO training."""
    def __init__(self, data_dir, split='train', transform=None, cache=False):
        """Initialize Dataset3D."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transform
        self.cache = cache
        self.data = []
        self.labels = []
        
        # Get volume files
        self.volume_files = list(self.data_dir.glob('**/*.npy'))
        if not self.volume_files:
            raise FileNotFoundError(f'No volume files found in {self.data_dir}')
        
        # Get labels
        self.labels_info = self.get_labels()
        
        # Cache data if requested
        if self.cache:
            self.cache_labels()
            
        LOGGER.info(f'Found {len(self.volume_files)} volume files in {self.data_dir}')

    def cache_labels(self):
        """Cache dataset labels for faster training."""
        LOGGER.info(f'Caching {len(self.volume_files)} volumes and labels...')
        self.data = []
        self.labels = []
        for volume_path, label_info in TQDM(zip(self.volume_files, self.labels_info), desc="Caching..."):
            volume = np.load(volume_path)
            self.data.append(torch.from_numpy(volume))
            self.labels.append(label_info)

    def get_labels(self):
        """Returns dictionary of labels for training."""
        labels = []
        for volume_path in self.volume_files:
            label_path = Path(str(volume_path).replace("/volumes/", "/labels/")).with_suffix('.txt')
            
            if label_path.exists():
                # Load and parse labels
                with open(label_path) as f:
                    bboxes = []
                    for line in f:
                        # Parse label format: class x y z width height depth
                        values = list(map(float, line.strip().split()))
                        bboxes.append(values)
                    bboxes = np.array(bboxes)
            else:
                bboxes = np.zeros((0, 7))  # Empty array with correct shape
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 7))  # Empty array with correct shape

                
            labels.append({
                'im_file': str(volume_path),
                'volume_file': str(volume_path),
                'label_file': str(label_path),
                'cls': bboxes[:, 0:1],  # n, 1
                'bboxes': bboxes[:, 1:],  # n, 6 (x,y,z,w,h,d)
                'normalized': True,
                'bbox_format': 'xyzwhd',
            })
        return labels

    def __len__(self):
        """Return the total number of samples."""
        return len(self.volume_files)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.cache:
            volume = self.data[idx]
            label_info = self.labels[idx]
        else:
            volume_path = self.volume_files[idx]
            volume = torch.from_numpy(np.load(volume_path))
            label_info = self.labels_info[idx]
            
        # Combine cls and bboxes for the final labels
        if len(label_info['cls']):
            labels = np.concatenate([label_info['cls'], label_info['bboxes']], axis=1)
        else:
            labels = np.zeros((0, 7))  # Empty array with correct shape
            
        # Apply transforms if any
        if self.transforms:
            volume, labels = self.transforms(volume, labels)
            
        return {
            'img': volume,  # Using 'img' key to maintain compatibility with YOLOv8
            'cls': torch.from_numpy(labels[:, 0:1]) if len(labels) else torch.zeros((0, 1)),
            'bboxes': torch.from_numpy(labels[:, 1:]) if len(labels) else torch.zeros((0, 6)),
            'path': label_info['volume_file'],
            'batch_idx': torch.zeros(len(labels)) if len(labels) else torch.zeros(0),
            'ori_shape': volume.shape
        }

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"bboxes", "cls"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
