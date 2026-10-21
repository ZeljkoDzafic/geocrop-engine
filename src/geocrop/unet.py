"""Compact U-Net (Ronneberger et al., 2015) for field-boundary probability. Requires `pip install geocrop-engine[deep]`."""
import torch
from torch import nn
from torch.nn import functional as F


def _block(a: int, b: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Three-level encoder/decoder (input sizes divisible by 8), ~0.5 M parameters at width 16."""

    def __init__(self, channels: int, width: int = 16):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3 = _block(channels, w), _block(w, 2 * w), _block(2 * w, 4 * w)
        self.mid = _block(4 * w, 8 * w)
        self.d3, self.d2, self.d1 = _block(12 * w, 4 * w), _block(6 * w, 2 * w), _block(3 * w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        m = self.mid(F.max_pool2d(c, 2))
        u = self.d3(torch.cat([F.interpolate(m, size=c.shape[-2:], mode="bilinear", align_corners=False), c], 1))
        u = self.d2(torch.cat([F.interpolate(u, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        u = self.d1(torch.cat([F.interpolate(u, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1))
        return self.out(u)
