"""
Shared Transformer and Optimal Transport components for Line Matcher.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Perform Sinkhorn optimal transport normalization in log-space."""
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=-1)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """
    Log-Optimal Transport with dustbin row and column.
    scores: (B, M, N)
    Returns: (B, M+1, N+1) log-assignment matrix
    """
    b, m, n = scores.shape
    device = scores.device

    # Add dustbin row and column
    bin_score = scores.new_zeros(b, 1, 1)
    top = torch.cat([scores, scores.new_zeros(b, m, 1)], dim=-1)
    bottom = torch.cat([scores.new_zeros(b, 1, n), bin_score], dim=-1)
    Z = torch.cat([top, bottom], dim=1)  # (B, M+1, N+1)

    # Marginals
    norm_m = float(m)
    norm_n = float(n)
    log_mu = torch.cat([torch.zeros(m, device=device) - math.log(norm_m + 1.0),
                        torch.tensor([math.log(norm_n / (norm_m + norm_n + 1e-6))], device=device)]).repeat(b, 1)
    log_nu = torch.cat([torch.zeros(n, device=device) - math.log(norm_n + 1.0),
                        torch.tensor([math.log(norm_m / (norm_m + norm_n + 1e-6))], device=device)]).repeat(b, 1)

    return log_sinkhorn_iterations(Z, log_mu, log_nu, iters=iters)


class LineGeometryEncoder(nn.Module):
    """
    Encodes 2D line segment geometry into a continuous embedding.
    Line: [x1, y1, x2, y2] in image pixel coordinates.
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, lines: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        lines: (B, N, 4) [x1, y1, x2, y2]
        image_size: (H, W)
        Returns: (B, N, d_model)
        """
        H, W = image_size
        diag = math.hypot(H, W)

        x1 = lines[..., 0]
        y1 = lines[..., 1]
        x2 = lines[..., 2]
        y2 = lines[..., 3]

        cx = (x1 + x2) / (2.0 * W)  # Normalized [0, 1]
        cy = (y1 + y2) / (2.0 * H)

        dx = x2 - x1
        dy = y2 - y1
        length = torch.hypot(dx, dy) / diag

        theta = torch.atan2(dy, dx)
        sin2 = torch.sin(2.0 * theta)
        cos2 = torch.cos(2.0 * theta)

        geom = torch.stack([cx, cy, length, sin2, cos2], dim=-1)  # (B, N, 5)
        return self.mlp(geom)


class LineVisualSampler(nn.Module):
    """
    Uniformly samples points along 2D line segments from a dense feature map.
    """

    def __init__(self, in_channels: int = 512, d_model: int = 128, num_samples: int = 7) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, feature_map: torch.Tensor, lines: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        B, C, H_f, W_f = feature_map.shape
        _, N, _ = lines.shape
        device = feature_map.device
        H, W = image_size

        if N == 0:
            return torch.zeros((B, 0, self.proj[-2].out_features), device=device)

        K = self.num_samples
        ts = torch.linspace(0.0, 1.0, K, device=device).view(1, 1, K, 1)
        p1 = lines[..., 0:2].unsqueeze(2)
        p2 = lines[..., 2:4].unsqueeze(2)
        pts = (1.0 - ts) * p1 + ts * p2

        gx = (pts[..., 0] / (W - 1.0)) * 2.0 - 1.0
        gy = (pts[..., 1] / (H - 1.0)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        grid = grid.view(B, N * K, 1, 2)

        sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).view(B, N, K, C)
        pooled = torch.mean(sampled, dim=2)
        return self.proj(pooled)


class MultiHeadAttentionBlock(nn.Module):
    """Transformer block with Multi-Head Attention, residual connections, and MLP."""

    def __init__(self, d_model: int = 128, nhead: int = 4, dim_feedforward: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if desc0.shape[1] > 0:
            q0 = desc0
            k0 = desc0
            v0 = desc0
            attn_out0, _ = self.self_attn(q0, k0, v0)
            desc0 = self.norm1(desc0 + attn_out0)

        if desc1.shape[1] > 0:
            q1 = desc1
            k1 = desc1
            v1 = desc1
            attn_out1, _ = self.self_attn(q1, k1, v1)
            desc1 = self.norm1(desc1 + attn_out1)

        if desc0.shape[1] > 0 and desc1.shape[1] > 0:
            cross_out0, _ = self.cross_attn(desc0, desc1, desc1)
            cross_out1, _ = self.cross_attn(desc1, desc0, desc0)
            desc0 = self.norm2(desc0 + cross_out0)
            desc1 = self.norm2(desc1 + cross_out1)

        if desc0.shape[1] > 0:
            ffn0 = self.linear2(self.dropout(F.relu(self.linear1(desc0))))
            desc0 = self.norm3(desc0 + ffn0)

        if desc1.shape[1] > 0:
            ffn1 = self.linear2(self.dropout(F.relu(self.linear1(desc1))))
            desc1 = self.norm3(desc1 + ffn1)

        return desc0, desc1
