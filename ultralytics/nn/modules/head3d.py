# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""3D Detection head modules."""

import torch
import torch.nn as nn
from .conv3d import Conv3d, DWConv3d

class Detect3D(nn.Module):
    """YOLOv8 3D Detect head for detection models."""
    
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initialize 3D detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 6  # number of outputs per anchor (6 for x,y,z,d,h,w)
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 6)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv3d(x, c2, 3), Conv3d(c2, c2, 3), nn.Conv3d(c2, 6 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv3d(x, x, 3), Conv3d(x, c3, 1)),
                nn.Sequential(DWConv3d(c3, c3, 3), Conv3d(c3, c3, 1)),
                nn.Conv3d(c3, self.nc, 1),
            )
            for x in ch
        )

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        
        # Add 3D detection specific post-processing here
        return x 