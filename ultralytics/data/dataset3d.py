# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import LOGGER, TQDM

class LoadVolumetricData(Dataset):
    """YOLOv8 3D Dataset for volumetric data."""
    
    def __init__(self, path, imgsz=640, cache=False):
        """Initialize 3D Dataset."""
        super().__init__()
        self.path = Path(path)
        self.imgsz = imgsz
        self.cache = cache
        self.data = []
        self.labels = []
        
        # Load data paths and labels
        self.volume_files = sorted([x for x in self.path.rglob('*.npy') if x.is_file()])  # adjust extension as needed
        self.label_files = [x.with_suffix('.txt') for x in self.volume_files]
        
        # Cache data
        if cache:
            self.cache_data()
            
    def cache_data(self):
        """Cache dataset for faster training."""
        LOGGER.info(f'Caching {len(self.volume_files)} 3D volumes...')
        self.data = []
        self.labels = []
        for f in TQDM(self.volume_files):
            volume = np.load(f)  # load 3D volume
            self.data.append(torch.from_numpy(volume))
            
            # Load labels (x, y, z, d, h, w, class)
            label_file = f.with_suffix('.txt')
            if label_file.exists():
                labels = np.loadtxt(label_file)
                labels = torch.from_numpy(labels)
            else:
                labels = torch.zeros((0, 7))
            self.labels.append(labels)
            
    def __len__(self):
        """Return the total number of samples."""
        return len(self.volume_files)
    
    def __getitem__(self, index):
        """Get a sample and its labels."""
        if self.cache:
            volume = self.data[index]
            labels = self.labels[index]
        else:
            volume_path = self.volume_files[index]
            volume = torch.from_numpy(np.load(volume_path))
            
            label_path = self.label_files[index]
            if label_path.exists():
                labels = torch.from_numpy(np.loadtxt(label_path))
            else:
                labels = torch.zeros((0, 7))
                
        # Preprocessing
        volume = self.preprocess_volume(volume)
        labels = self.preprocess_labels(labels)
        
        return volume, labels
    
    def preprocess_volume(self, volume):
        """Preprocess 3D volume."""
        # Normalize
        volume = volume.float() / 255.0
        
        # Resize if needed
        if volume.shape[1:] != (self.imgsz, self.imgsz, self.imgsz):
            volume = torch.nn.functional.interpolate(
                volume.unsqueeze(0),
                size=(self.imgsz, self.imgsz, self.imgsz),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
            
        return volume
    
    def preprocess_labels(self, labels):
        """Preprocess labels."""
        # Convert absolute coordinates to relative
        if len(labels):
            labels[:, [0, 1, 2]] = labels[:, [0, 1, 2]] / self.imgsz
            labels[:, [3, 4, 5]] = labels[:, [3, 4, 5]] / self.imgsz
        return labels 