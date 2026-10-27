"""Along-line texture encoder: sample CNN feature-map pixels along each line."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def endpoints_from_5d(lines_5d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover endpoints in patch-normalized [0, 1] coords from 5D descriptors
    [cu, cv, length, sin(2θ), cos(2θ)].

    Returns:
        p1, p2: each (..., 2) as (u, v).
    """
    cu = lines_5d[..., 0]
    cv = lines_5d[..., 1]
    length = lines_5d[..., 2].clamp(min=0.0)
    sin2 = lines_5d[..., 3]
    cos2 = lines_5d[..., 4]
    # 2θ -> θ in [0, π); atan2(sin2, cos2) / 2
    theta = 0.5 * torch.atan2(sin2, cos2)
    du = 0.5 * length * torch.cos(theta)
    dv = 0.5 * length * torch.sin(theta)
    p1 = torch.stack([cu - du, cv - dv], dim=-1)
    p2 = torch.stack([cu + du, cv + dv], dim=-1)
    return p1, p2


class AlongLineTextureEncoder(nn.Module):
    """
    For each line slot, sample `num_samples` points along the segment on a feature map
    via grid_sample, then aggregate with Conv1d + mean pool to a line vector.
    """

    def __init__(self, in_channels: int, out_dim: int = 64, num_samples: int = 8) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.num_samples = int(num_samples)

        self.conv1d = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.out_dim, self.out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
        )
        self.null_token = nn.Parameter(torch.zeros(self.out_dim))
        nn.init.normal_(self.null_token, std=0.02)

    def sample_along_lines(
        self,
        fmap: torch.Tensor,
        lines_5d: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        fmap: (B*M, C, Hf, Wf) or (N, C, Hf, Wf) — one map per region instance
        lines_5d: (N, K, 5) aligned with fmap batch dim N=B*M
        valid_mask: (N, K) bool

        Returns: (N, K, C) sampled-and-pooled features before out_proj
        (actually returns after conv1d mean: (N, K, out_dim) raw before out_proj)
        """
        N, C, Hf, Wf = fmap.shape
        K = lines_5d.shape[1]
        T = self.num_samples
        device = fmap.device
        dtype = fmap.dtype

        p1, p2 = endpoints_from_5d(lines_5d)  # (N, K, 2)
        # Sample positions in [0,1]
        ts = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).view(1, 1, T, 1)
        pts = p1.unsqueeze(2) * (1.0 - ts) + p2.unsqueeze(2) * ts  # (N, K, T, 2) as (u,v)
        pts = pts.clamp(0.0, 1.0)
        # grid_sample expects (x, y) in [-1, 1]
        grid = pts * 2.0 - 1.0  # (N, K, T, 2)
        # Reshape to sample K*T points per map: grid (N, K, T, 2) -> (N, K*T, 1, 2) then reshape
        grid_flat = grid.view(N, K * T, 1, 2)
        sampled = F.grid_sample(
            fmap,
            grid_flat,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (N, C, K*T, 1)
        sampled = sampled.squeeze(-1).view(N, C, K, T)  # (N, C, K, T)
        sampled = sampled.permute(0, 2, 1, 3).contiguous()  # (N, K, C, T)

        x = sampled.view(N * K, C, T)
        x = self.conv1d(x)  # (N*K, out_dim, T)
        x = x.mean(dim=-1)  # (N*K, out_dim)
        x = x.view(N, K, self.out_dim)

        if valid_mask is not None:
            vm = valid_mask.to(dtype=x.dtype).unsqueeze(-1)
            x = x * vm + self.null_token.view(1, 1, -1) * (1.0 - vm)
        return x

    def forward(
        self,
        fmap: torch.Tensor,
        lines_5d: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns line features (N, K, out_dim).
        """
        raw = self.sample_along_lines(fmap, lines_5d, valid_mask=valid_mask)
        return self.out_proj(raw)
