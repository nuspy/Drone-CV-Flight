"""Tiny convolutional backbone, width-scalable via config.

backbone_width=16 trains in CI on CPU at 64x64; 48+ is appropriate for real
Unity environments at 224x224 on GPU. GroupNorm keeps small-batch training
stable.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    import math

    groups = math.gcd(c_out, 8)
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
        nn.GroupNorm(groups, c_out),
        nn.SiLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, stride=1, padding=1),
        nn.GroupNorm(groups, c_out),
        nn.SiLU(inplace=True),
    )


class TinyConvNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(3, w),
            _block(w, 2 * w),
            _block(2 * w, 4 * w),
            _block(4 * w, 8 * w),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 8 * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)
