"""Along-line texture encoder (placeholder scaffold; filled in later steps)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AlongLineTextureEncoder(nn.Module):
    """Sample feature-map pixels along each line and aggregate to a line vector."""

    def __init__(self, in_channels: int, out_dim: int = 64, num_samples: int = 8) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.num_samples = int(num_samples)
        # Concrete layers added in step04.
        self._placeholder = nn.Identity()

    def forward(self, *args, **kwargs):  # pragma: no cover - filled in step04
        raise NotImplementedError("AlongLineTextureEncoder.forward is implemented in step04")
