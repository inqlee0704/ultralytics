# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""3D Convolution modules."""

import math
import torch
import torch.nn as nn

__all__ = (
    "Conv3d",
    "DWConv3d",
    "Conv3dTranspose",
    "Focus3d",
    "Concat3d",
)

def autopad3d(k, p=None, d=1):
    """Pad to 'same' shape outputs in 3D."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv3d(nn.Module):
    """Standard 3D convolution."""
    
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv3d layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, autopad3d(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # Print shapes for debugging
        # print(f"Conv3d shapes - Input: {x.shape}, Weight: {self.conv.weight.shape}")
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform forward pass through Conv3d layer."""
        return self.act(self.conv(x))

class DWConv3d(Conv3d):
    """3D Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize DWConv3d layer with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class Conv3dTranspose(nn.Module):
    """3D Convolution transpose layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose3d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm3d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

class Focus3d(nn.Module):
    """Focus wh information into c-space for 3D data."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus3d layer."""
        super().__init__()
        self.conv = Conv3d(c1 * 8, c2, k, s, p, g, act=act)

    def forward(self, x):
        """Forward pass through Focus3d layer."""
        # Assuming x shape is (b,c,d,h,w)
        return self.conv(torch.cat([
            x[..., ::2, ::2, ::2],
            x[..., 1::2, ::2, ::2],
            x[..., ::2, 1::2, ::2],
            x[..., ::2, ::2, 1::2],
            x[..., 1::2, 1::2, ::2],
            x[..., 1::2, ::2, 1::2],
            x[..., ::2, 1::2, 1::2],
            x[..., 1::2, 1::2, 1::2]
        ], 1))

class Concat3d(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Initialize Concat3d layer."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass through Concat3d layer."""
        return torch.cat(x, self.d) 