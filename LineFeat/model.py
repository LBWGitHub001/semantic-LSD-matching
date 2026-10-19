"""
LineFeat Trainable Line Matching Network (Line-Transformer).

Architecture:
  1. Dense Feature Map Extractor (CNN Backbone)
  2. Line Feature Sampler (continuous grid sampling along line segments)
  3. Geometric Position Encoder (normalized center, length, sin2θ, cos2θ)
  4. Line Transformer (alternating Multi-Head Self-Attention & Cross-Attention)
  5. Optimal Transport Head (Log-Sinkhorn with Dustbin)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
    # Augment scores with dustbins
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
    Attributes: normalized center (cx, cy), length l, and orientation (sin 2θ, cos 2θ).
    Using 2θ naturally eliminates the 180-degree direction ambiguity.
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        # Input features: [cx, cy, length, sin(2θ), cos(2θ)] -> 5 dimensions
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
        """
        feature_map: (B, C, H_f, W_f)
        lines: (B, N, 4) [x1, y1, x2, y2]
        image_size: (H, W)
        Returns: (B, N, d_model)
        """
        B, C, H_f, W_f = feature_map.shape
        _, N, _ = lines.shape
        device = feature_map.device
        H, W = image_size

        if N == 0:
            return torch.zeros((B, 0, self.proj[-2].out_features), device=device)

        # 1. Sample K points along line segment
        K = self.num_samples
        ts = torch.linspace(0.0, 1.0, K, device=device).view(1, 1, K, 1)  # (1, 1, K, 1)
        p1 = lines[..., 0:2].unsqueeze(2)  # (B, N, 1, 2)
        p2 = lines[..., 2:4].unsqueeze(2)  # (B, N, 1, 2)
        pts = (1.0 - ts) * p1 + ts * p2  # (B, N, K, 2) [x, y]

        # 2. Normalize to [-1, 1] for grid_sample
        gx = (pts[..., 0] / (W - 1.0)) * 2.0 - 1.0
        gy = (pts[..., 1] / (H - 1.0)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1)  # (B, N, K, 2)
        grid = grid.view(B, N * K, 1, 2)  # (B, N*K, 1, 2)

        # 3. Bilinear interpolation
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
        # sampled: (B, C, N*K, 1) -> (B, N, K, C)
        sampled = sampled.squeeze(-1).permute(0, 2, 1).view(B, N, K, C)

        # 4. Order-invariant pooling (Mean pooling across K points)
        pooled = torch.mean(sampled, dim=2)  # (B, N, C)

        # 5. Project to d_model
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
        """
        desc0: (B, N0, D)
        desc1: (B, N1, D)
        """
        # 1. Self-Attention within each image
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

        # 2. Cross-Attention between images
        if desc0.shape[1] > 0 and desc1.shape[1] > 0:
            cross_out0, _ = self.cross_attn(desc0, desc1, desc1)
            cross_out1, _ = self.cross_attn(desc1, desc0, desc0)
            desc0 = self.norm2(desc0 + cross_out0)
            desc1 = self.norm2(desc1 + cross_out1)

        # 3. FFN
        if desc0.shape[1] > 0:
            ffn0 = self.linear2(self.dropout(F.relu(self.linear1(desc0))))
            desc0 = self.norm3(desc0 + ffn0)

        if desc1.shape[1] > 0:
            ffn1 = self.linear2(self.dropout(F.relu(self.linear1(desc1))))
            desc1 = self.norm3(desc1 + ffn1)

        return desc0, desc1


class LineMatcherNet(nn.Module):
    """
    Complete End-to-End Line Matching Network for already detected lines.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        nhead: int = 4,
        sinkhorn_iters: int = 20,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters

        # Backbone: Pretrained ResNet50 up to layer2 (512 channels)
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.sampler = LineVisualSampler(in_channels=512, d_model=d_model, num_samples=7)
        self.geometry_encoder = LineGeometryEncoder(d_model=d_model)

        # Alternating Transformer Attention Blocks
        self.layers = nn.ModuleList([
            MultiHeadAttentionBlock(d_model=d_model, nhead=nhead)
            for _ in range(num_layers)
        ])

        # Final descriptor projections
        self.proj0 = nn.Linear(d_model, d_model)
        self.proj1 = nn.Linear(d_model, d_model)

    def extract_line_tokens(
        self,
        image: torch.Tensor,
        lines: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract line tokens combining visual feature sampling and geometric encoding.
        image: (B, 3, H, W)
        lines: (B, N, 4)
        """
        B, _, H, W = image.shape
        feat_map = self.backbone(image)  # (B, 512, H/8, W/8)
        vis_emb = self.sampler(feat_map, lines, (H, W))  # (B, N, D)
        geom_emb = self.geometry_encoder(lines, (H, W))  # (B, N, D)
        tokens = vis_emb + geom_emb
        return tokens

    def forward(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        lines0: torch.Tensor,
        lines1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image0, image1: (B, 3, H, W)
            lines0: (B, N0, 4)
            lines1: (B, N1, 4)

        Returns:
            dict containing:
              - 'log_assignment': (B, N0+1, N1+1) Log-Sinkhorn probability matrix
              - 'scores': (B, N0, N1) Cosine similarity matrix
              - 'desc0', 'desc1': (B, N, D) Final line descriptors
        """
        tokens0 = self.extract_line_tokens(image0, lines0)
        tokens1 = self.extract_line_tokens(image1, lines1)

        # Transformer layers
        desc0, desc1 = tokens0, tokens1
        for layer in self.layers:
            desc0, desc1 = layer(desc0, desc1)

        desc0 = F.normalize(self.proj0(desc0), p=2, dim=-1)
        desc1 = F.normalize(self.proj1(desc1), p=2, dim=-1)

        # Match scores (B, N0, N1)
        scores = torch.bmm(desc0, desc1.transpose(1, 2))

        # Optimal transport
        log_P = log_optimal_transport(scores, iters=self.sinkhorn_iters)

        return {
            "log_assignment": log_P,
            "scores": scores,
            "desc0": desc0,
            "desc1": desc1,
        }

    def forward_from_vis_tokens(
        self,
        vis0: torch.Tensor,
        vis1: torch.Tensor,
        lines0: torch.Tensor,
        lines1: torch.Tensor,
        image_size: Tuple[int, int] = (720, 1280),
    ) -> dict[str, torch.Tensor]:
        """
        Fast forward pass directly from precomputed visual embeddings.
        Bypasses CNN backbone and image loading entirely.
        """
        geom0 = self.geometry_encoder(lines0, image_size)
        geom1 = self.geometry_encoder(lines1, image_size)
        tokens0 = vis0 + geom0
        tokens1 = vis1 + geom1

        desc0, desc1 = tokens0, tokens1
        for layer in self.layers:
            desc0, desc1 = layer(desc0, desc1)

        desc0 = F.normalize(self.proj0(desc0), p=2, dim=-1)
        desc1 = F.normalize(self.proj1(desc1), p=2, dim=-1)

        scores = torch.bmm(desc0, desc1.transpose(1, 2))
        log_P = log_optimal_transport(scores, iters=self.sinkhorn_iters)

        return {
            "log_assignment": log_P,
            "scores": scores,
            "desc0": desc0,
            "desc1": desc1,
        }
