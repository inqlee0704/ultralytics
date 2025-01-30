# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""3D Detection head modules."""

import torch
import torch.nn as nn
from .conv3d import Conv3d, DWConv3d
from .block3d import DFL3d
import math
from ultralytics.utils.tal import make_anchors_3d

class Detect3D(nn.Module):
    """YOLO 3D Detect head for detection models."""
    
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=6, ch=()):
        """Initialize 3D detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 6  # number of outputs per anchor (6 for x,y,z,d,h,w)
        # self.stride = torch.zeros(self.nl)  # strides computed during build
        self.stride = torch.tensor([8, 16, 32])  # strides computed during build
        
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
        self.dfl = DFL3d(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        
        self.stride = torch.tensor([8, 16, 32]).to(x[0].device, dtype=torch.float16)  # strides computed during build
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors_3d(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 6]  # 6 box parameters for 3D
            cls = x_cat[:, self.reg_max * 6:]
        else:
            box, cls = x_cat.split((self.reg_max * 6, self.nc), 1)  # 6 box parameters for 3D

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            grid_h = shape[2]
            grid_w = shape[3]
            grid_d = shape[4]  # Added depth dimension
            grid_size = torch.tensor([grid_w, grid_h, grid_d, grid_w, grid_h, grid_d], device=box.device).reshape(1, 6, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :3])  # Only scale x,y,z
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def decode_bboxes(self, bboxes, anchors):
        """Decode 3D bounding boxes."""
        # Split predictions into xy, z, and dimensions
        xy, z, dims = bboxes.split([2, 1, 3], dim=1)
        # Decode xy similar to 2D case
        xy = (xy * 2 + anchors[:, :2]) * self.strides
        # Decode z
        z = (z * 2 + anchors[:, 2:3]) * self.strides
        # Decode dimensions (d, h, w)
        dims = (dims * 2) ** 2 * self.strides
        # Combine all predictions
        return torch.cat((xy, z, dims), dim=1)

    def bias_init(self):
        """Initialize Detect3D() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box (6 parameters for 3D)
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img) 