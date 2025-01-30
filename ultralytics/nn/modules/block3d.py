# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
from .conv3d import Conv3d

class DFL3d(nn.Module):
    """
    3D version of the Integral module of Distribution Focal Loss (DFL).
    
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    Adapted for 3D operations with 6 coordinates (x, y, z, width, height, depth).
    """

    def __init__(self, c1=16):
        """Initialize a 3D convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv3d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """
        Applies DFL operation on input tensor 'x'.
        
        Args:
            x: Input tensor with shape (batch, channels, anchors)
               where channels = 6 * c1 for 3D boxes (x, y, z, w, h, d)
        
        Returns:
            Tensor with shape (batch, 6, anchors) containing box coordinates
        """
        b, _, a = x.shape  # batch, channels, anchors
        # Reshape to handle 6 coordinates (x,y,z,w,h,d) instead of 4 (x,y,w,h)
        x = x.view(b, 6, self.c1, 1, a)
        x = x.softmax(2).transpose(2, 1)
        return self.conv(x).view(b, 6, a)
        # return self.conv(x.view(b, 6, self.c1, a).transpose(2, 1).softmax(1)).view(b, 6, a)

class C3_3d(nn.Module):
    """3D version of CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv3d(c1, c_, 1, 1)
        self.cv2 = Conv3d(c1, c_, 1, 1)
        self.cv3 = Conv3d(2 * c_, c2, 1)  # Ensure output channels match
        self.m = nn.Sequential(*(Bottleneck3d(c_, c_, shortcut, g, k=((1, 1, 1), (3, 3, 3))) for _ in range(n)))

    def forward(self, x):
        """Forward pass through C3 module."""
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        # Print shapes for debugging
        print(f"C3_3d shapes - Input: {x.shape}, cv1: {y1.shape}, cv2: {y2.shape}")
        return self.cv3(torch.cat((y1, y2), 1))

class Bottleneck3d(nn.Module):
    """3D version of Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=((1, 1, 1), (3, 3, 3))):
        super().__init__()
        c_ = c2 if g == 1 else c2 // 2
        self.cv1 = Conv3d(c1, c_, k[0], 1)
        self.cv2 = Conv3d(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass through bottleneck."""
        # Print shapes for debugging
        # print(f"Bottleneck3d shapes - Input: {x.shape}")
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SPPF3d(nn.Module):
    """3D version of Spatial Pyramid Pooling - Fast (SPPF) layer."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv3d(c1, c_, 1, 1)
        self.cv2 = Conv3d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through SPPF layer."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class C3k_3d(C3_3d):
    """3D version of C3k - CSP bottleneck module with customizable kernel sizes for 3D feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k_3d module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck3d(c_, c_, shortcut, g, k=((k,k,k), (k,k,k))) for _ in range(n)))

class C2f_3d(nn.Module):
    """3D version of Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv3d(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv3d((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck3d(self.c, self.c, shortcut, g, k=((3, 3, 3), (3, 3, 3))) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f_3d layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3k2_3d(C2f_3d):
    """3D version of C3k2 - Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2_3d module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_3d(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck3d(self.c, self.c, shortcut, g) for _ in range(n)
        )

class C3k_3d(nn.Module):
    """3D version of C3k module."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initialize C3k_3d module with given parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv3d(c1, c_, 1, 1)
        self.cv2 = Conv3d(c1, c_, 1, 1)
        self.cv3 = Conv3d(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck3d(c_, c_, shortcut, g, k=((k, k, k), (k, k, k))) for _ in range(n)))

    def forward(self, x):
        """Forward pass through C3k_3d module."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)) 

class Attention3d(nn.Module):
    """
    Attention module that performs self-attention on the 3D input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv3d): 3D convolutional layer for computing the query, key, and value.
        proj (Conv3d): 3D convolutional layer for projecting the attended values.
        pe (Conv3d): 3D convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv3d(dim, h, 1, act=False)
        self.proj = Conv3d(dim, dim, 1, act=False)
        self.pe = Conv3d(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, D, H, W = x.shape
        N = D * H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, D, H, W) + self.pe(v.reshape(B, C, D, H, W))
        x = self.proj(x)
        return x


class PSABlock3d(nn.Module):
    """
    PSABlock3d class implementing a Position-Sensitive Attention block for 3D neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections in 3D.

    Attributes:
        attn (Attention3d): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock3d, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock3d and perform a forward pass
        >>> psablock = PSABlock3d(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock3d with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention3d(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv3d(c, c * 2, 1), Conv3d(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock3d, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA3d(nn.Module):
    """
    C2PSA3d module with attention mechanism for enhanced 3D feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities in 3D. It includes a series of PSABlock3d modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv3d): 1x1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv3d): 1x1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock3d modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA3d module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA3d module, but refactored to allow stacking more PSABlock3d modules.

    Examples:
        >>> c2psa = C2PSA3d(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA3d module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv3d(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv3d(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock3d(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))