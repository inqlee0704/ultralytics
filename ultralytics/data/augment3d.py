# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class Augment3D:
    """3D data augmentation class."""
    def __init__(self, size=640, scale=(0.8, 1.2), translate=0.2, flip_prob=0.0, rotate=15):
        """Initialize 3D augmentation parameters."""
        self.size = size
        self.scale = scale
        self.translate = translate
        self.flip_prob = 0.0
        # self.flip_prob = flip_prob
        self.rotate = rotate

    def __call__(self, volume, labels=None):
        """Apply augmentation to volume and labels."""
        # Convert to tensor if needed
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
        
        # Add batch dimension if needed
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        # Random scaling
        # scale = np.random.uniform(self.scale[0], self.scale[1])
        # new_size = int(self.size * scale)
        # volume = F.interpolate(volume.unsqueeze(0), size=(new_size, new_size, new_size), 
                            #  mode='trilinear', align_corners=True)[0]
        
        # Random translation
        # if self.translate > 0:
        #     tx = int(self.translate * new_size * (np.random.random() - 0.5))
        #     ty = int(self.translate * new_size * (np.random.random() - 0.5))
        #     tz = int(self.translate * new_size * (np.random.random() - 0.5))
        #     volume = torch.roll(volume, shifts=(tx, ty, tz), dims=(0, 1, 2))
        
        # Random flipping
        if np.random.random() < self.flip_prob:
            volume = volume.flip(2)  # flip along z-axis
            if labels is not None:
                # Adjust labels for flipped volume
                labels[:, 2] = 1 - labels[:, 2]  # flip z coordinates
        
        # Random rotation (currently only around z-axis)
        # if self.rotate > 0:
            # angle = np.random.uniform(-self.rotate, self.rotate)
            # Implement 3D rotation here
            # volume = rotate_volume(volume, angle)
        # Resize to target size if needed
        # if volume.shape[-1] != self.size:
        #     volume = F.interpolate(volume.unsqueeze(0), size=(self.size, self.size, self.size),
        #                          mode='trilinear', align_corners=True)[0]
        
        # Normalize volume
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
        
        return volume, labels


class Compose3D:
    """Compose multiple 3D transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, volume, labels=None):
        for t in self.transforms:
            volume, labels = t(volume, labels)
        return volume, labels 