"""
Geometry-Field Prompted & Mask-Pooled Line Matching Network (MaskLineMatcherNet).

Architecture:
  1. Input: (B, 6, H, W) dual tensors combining RGB images with continuous geometry fields (EDT + AFM).
  2. 6-Channel Feature Extractor: ResNet-50 backbone initialized with pretrained weights.
  3. Mask-Aware Line Tokenizer: Extracts visual-geometry line tokens for detected line segments.
  4. Cross-Image Line-Transformer: 4 alternating layers of Self-Attention and Cross-Attention.
  5. Optimal Transport (Sinkhorn): Learns assignment with Dustbin bin to reject unmatched lines.
  6. Canonical 255x255 Probability Matrix: Dynamically populated affinity matrix matching user's canonical interface.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from LineFeat.geometric_field import create_6ch_input_tensor
from LineFeat.model import (
    LineGeometryEncoder,
    LineVisualSampler,
    MultiHeadAttentionBlock,
    log_optimal_transport,
)


class MaskLineMatcherNet(nn.Module):
    """
    End-to-end 6-channel Geometry-Field prompted Line Matching Network.

    Takes two 6-channel input representations (RGB + EDT + AFM) alongside detected line coordinates,
    and directly outputs a canonical (B, 255, 255) match probability matrix.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        nhead: int = 4,
        sinkhorn_iters: int = 20,
        max_matrix_size: int = 255,
        freeze_backbone_stages: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters
        self.max_matrix_size = max_matrix_size

        # 1. 6-Channel ResNet-50 Backbone
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Expand first conv from 3 to 6 channels
        conv1_6ch = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Copy RGB pretrained weights to first 3 channels
            conv1_6ch.weight[:, :3] = base.conv1.weight
            # Initialize geometry channels (EDT + AFM) from average RGB filters
            conv1_6ch.weight[:, 3:] = base.conv1.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        self.backbone = nn.Sequential(
            conv1_6ch,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,  # 512 channels, stride 4
        )

        if freeze_backbone_stages >= 1:
            for p in self.backbone[0:5].parameters():
                p.requires_grad = False

        # 2. Line Tokenizer (Sampler + Geometric Encoder)
        self.sampler = LineVisualSampler(in_channels=512, d_model=d_model, num_samples=7)
        self.geometry_encoder = LineGeometryEncoder(d_model=d_model)

        # 3. Transformer Layers
        self.layers = nn.ModuleList([
            MultiHeadAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2)
            for _ in range(num_layers)
        ])

        # 4. Final Projection Heads
        self.proj0 = nn.Linear(d_model, d_model)
        self.proj1 = nn.Linear(d_model, d_model)

    def extract_features(self, x_6ch: torch.Tensor) -> torch.Tensor:
        """Runs 6-channel input through CNN backbone."""
        return self.backbone(x_6ch)

    def forward(
        self,
        input0: torch.Tensor,
        input1: torch.Tensor,
        lines0: torch.Tensor,
        lines1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input0: (B, 6, H, W) or (B, 3, H, W) for View 0.
                    If 3 channels, continuous geometry field will be auto-generated.
            input1: (B, 6, H, W) or (B, 3, H, W) for View 1.
            lines0: (B, N0, 4) coordinates [x1, y1, x2, y2].
            lines1: (B, N1, 4) coordinates [x1, y1, x2, y2].

        Returns:
            Dictionary containing:
              - 'matrix_255': (B, 255, 255) Canonical match probability matrix.
              - 'log_assignment': (B, N0+1, N1+1) Log-Sinkhorn probability distribution.
              - 'P_sub': (B, N0, N1) Submatrix probability of mutual matches.
              - 'scores': (B, N0, N1) Raw pairwise cosine similarity scores.
              - 'desc0', 'desc1': L2-normalized line descriptors.
        """
        B = lines0.shape[0]
        device = input0.device

        # Auto-expand to 6 channels if 3-channel RGB provided
        if input0.shape[1] == 3:
            input0 = create_6ch_input_tensor(input0, lines0)
        if input1.shape[1] == 3:
            input1 = create_6ch_input_tensor(input1, lines1)

        H, W = input0.shape[2], input0.shape[3]
        N0 = lines0.shape[1]
        N1 = lines1.shape[1]

        # 1. CNN Feature Maps: (B, 512, H/4, W/4)
        feat0 = self.extract_features(input0)
        feat1 = self.extract_features(input1)

        # 2. Tokenize Line Segments (Visual + Geometry)
        vis0 = self.sampler(feat0, lines0, (H, W))  # (B, N0, D)
        vis1 = self.sampler(feat1, lines1, (H, W))  # (B, N1, D)

        geom0 = self.geometry_encoder(lines0, (H, W))
        geom1 = self.geometry_encoder(lines1, (H, W))

        tokens0 = vis0 + geom0
        tokens1 = vis1 + geom1

        # 3. Cross-Image Line-Transformer Reasoning
        desc0, desc1 = tokens0, tokens1
        for layer in self.layers:
            desc0, desc1 = layer(desc0, desc1)

        desc0 = F.normalize(self.proj0(desc0), p=2, dim=-1)
        desc1 = F.normalize(self.proj1(desc1), p=2, dim=-1)

        # 4. Pairwise Cosine Similarity: (B, N0, N1)
        scores = torch.bmm(desc0, desc1.transpose(1, 2))

        # 5. Optimal Transport (Sinkhorn with Dustbin)
        log_P = log_optimal_transport(scores, iters=self.sinkhorn_iters)
        P = torch.exp(log_P)  # (B, N0+1, N1+1)
        P_sub = P[:, :N0, :N1]  # (B, N0, N1)

        # 6. Build Canonical (B, 255, 255) Probability Matrix
        M_max = self.max_matrix_size
        matrix_255 = torch.zeros((B, M_max, M_max), device=device, dtype=torch.float32)

        valid_n0 = min(N0, M_max)
        valid_n1 = min(N1, M_max)

        if valid_n0 > 0 and valid_n1 > 0:
            matrix_255[:, :valid_n0, :valid_n1] = P_sub[:, :valid_n0, :valid_n1]

        return {
            "matrix_255": matrix_255,
            "log_assignment": log_P,
            "P_sub": P_sub,
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
        image_size: tuple[int, int] = (720, 1280),
    ) -> dict[str, torch.Tensor]:
        """
        Fast forward pass directly from visual embeddings.
        Bypasses CNN backbone and image loading entirely.
        """
        B = lines0.shape[0]
        N0 = lines0.shape[1]
        N1 = lines1.shape[1]
        device = lines0.device

        geom0 = self.geometry_encoder(lines0, image_size)
        geom1 = self.geometry_encoder(lines1, image_size)

        if vis0.shape[-1] != self.d_model:
            if not hasattr(self, "_cache_proj"):
                self._cache_proj = nn.Linear(vis0.shape[-1], self.d_model).to(device)
            vis0 = self._cache_proj(vis0)
            vis1 = self._cache_proj(vis1)

        tokens0 = vis0 + geom0
        tokens1 = vis1 + geom1

        desc0, desc1 = tokens0, tokens1
        for layer in self.layers:
            desc0, desc1 = layer(desc0, desc1)

        desc0 = F.normalize(self.proj0(desc0), p=2, dim=-1)
        desc1 = F.normalize(self.proj1(desc1), p=2, dim=-1)

        scores = torch.bmm(desc0, desc1.transpose(1, 2))
        log_P = log_optimal_transport(scores, iters=self.sinkhorn_iters)
        P = torch.exp(log_P)
        P_sub = P[:, :N0, :N1]

        M_max = self.max_matrix_size
        matrix_255 = torch.zeros((B, M_max, M_max), device=device, dtype=torch.float32)
        valid_n0 = min(N0, M_max)
        valid_n1 = min(N1, M_max)
        if valid_n0 > 0 and valid_n1 > 0:
            matrix_255[:, :valid_n0, :valid_n1] = P_sub[:, :valid_n0, :valid_n1]

        return {
            "matrix_255": matrix_255,
            "log_assignment": log_P,
            "P_sub": P_sub,
            "scores": scores,
            "desc0": desc0,
            "desc1": desc1,
        }


# Official unified alias requested by user
MaskLineFeatMatchNet = MaskLineMatcherNet

