"""
XFeat-style Object-Level Bipartite Matcher (XFeatObjectMatcher).

A lightweight, high-speed neural network for object-to-object correspondence ("连连看")
across two camera views:
  1. XFeat-style Convolutional Backbone: Ultra-fast 4-stage CNN extracting 64-dim dense feature maps (stride 8).
  2. Masked Average Pooling (MaskPooling): Aggregates localized feature maps inside each object's mask.
  3. Object Spatial Geometry Encoder: Encodes normalized bounding box, centroid, and area layout.
  4. Object Graph Transformer: Alternating Intra-View Self-Attention and Cross-View Cross-Attention.
  5. Optimal Transport with Dustbin (Log-Sinkhorn): Learns 1-to-1 bipartite assignment while rejecting out-of-view objects.
  6. SuperGlue-style NLL Loss: Supervises co-visible matches and dustbin rejections.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. XFeat-style Lightweight Convolutional Backbone
# ==============================================================================
class BasicResidualBlock(nn.Module):
    """Simple 2-conv residual block with batch norm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class XFeatBackbone(nn.Module):
    """
    Ultra-lightweight CNN backbone inspired by XFeat (CVPR 2024).
    Produces a 64-dimensional feature map at stride 8.
    Parameters: ~0.25M (sub-millisecond inference).
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 64) -> None:
        super().__init__()
        # Stage 1: Stride 2 (H/2, W/2)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            BasicResidualBlock(32),
        )
        # Stage 2: Stride 2 (H/4, W/4)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicResidualBlock(64),
        )
        # Stage 3: Stride 2 (H/8, W/8)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            BasicResidualBlock(128),
        )
        # Stage 4: Channel projection to out_dim
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        Returns: (B, out_dim, H/8, W/8)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


# ==============================================================================
# 2. Mask Pooling & Object Spatial Geometry Encoder
# ==============================================================================
class MaskPooling(nn.Module):
    """
    Aggregates visual features inside object masks via masked average pooling.
    Handles downsampling and variable mask sizes.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, feat_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        feat_map: (B, C, H_f, W_f)
        masks: (B, M, H, W) binary/float masks
        Returns: (B, M, C) object appearance tokens
        """
        B, C, H_f, W_f = feat_map.shape
        B_m, M, H_m, W_m = masks.shape
        assert B == B_m

        # Downsample masks to feature map resolution
        if (H_m, W_m) != (H_f, W_f):
            masks_down = F.interpolate(masks, size=(H_f, W_f), mode="nearest")
        else:
            masks_down = masks

        # (B, M, H_f, W_f) -> (B, M, 1, H_f*W_f)
        m_flat = masks_down.view(B, M, 1, H_f * W_f)
        # (B, C, H_f, W_f) -> (B, 1, C, H_f*W_f)
        f_flat = feat_map.view(B, 1, C, H_f * W_f)

        # Masked sum: (B, M, C)
        num = (f_flat * m_flat).sum(dim=-1)
        den = m_flat.sum(dim=-1) + self.eps  # (B, M, 1)

        return num / den


class ObjectGeometryEncoder(nn.Module):
    """
    Encodes object 2D bounding boxes and center locations into layout tokens.
    Input features: [x1/W, y1/H, x2/W, y2/H, cx/W, cy/H, w/W, h/H, area/(W*H)] -> 9 dims
    """

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, bboxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        bboxes: (B, M, 4) in pixel coords [x1, y1, x2, y2]
        image_size: (H, W)
        Returns: (B, M, out_dim)
        """
        H, W = image_size
        x1 = bboxes[..., 0] / W
        y1 = bboxes[..., 1] / H
        x2 = bboxes[..., 2] / W
        y2 = bboxes[..., 3] / H
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = torch.clamp(x2 - x1, min=0.0)
        h = torch.clamp(y2 - y1, min=0.0)
        area = w * h

        geo_vec = torch.stack([x1, y1, x2, y2, cx, cy, w, h, area], dim=-1)
        return self.mlp(geo_vec)


# ==============================================================================
# 3. Object Graph Transformer (Self + Cross Attention)
# ==============================================================================
class AttentionBlock(nn.Module):
    """Multi-head attention block with pre-LayerNorm and MLP residual."""

    def __init__(self, d_model: int = 128, nhead: int = 4) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm1(query)
        k_norm = self.norm1(key)
        v_norm = self.norm1(value)
        attn_out, _ = self.mha(q_norm, k_norm, v_norm)
        x = query + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ObjectGraphTransformer(nn.Module):
    """
    Alternating Intra-View Self-Attention and Cross-View Cross-Attention layers.
    Allows objects within a view to communicate spatial context (e.g. Table next to Door),
    and objects across views to compare visual-topological similarities.
    """

    def __init__(self, d_model: int = 128, num_layers: int = 2, nhead: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Self-attention for View 0 and View 1
            self.layers.append(nn.ModuleDict({
                "self0": AttentionBlock(d_model, nhead),
                "self1": AttentionBlock(d_model, nhead),
                "cross0": AttentionBlock(d_model, nhead),
                "cross1": AttentionBlock(d_model, nhead),
            }))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x0: (B, M0, d_model)
        x1: (B, M1, d_model)
        Returns: (B, M0, d_model), (B, M1, d_model)
        """
        for layer in self.layers:
            # Intra-view self-attention
            x0 = layer["self0"](x0, x0, x0)
            x1 = layer["self1"](x1, x1, x1)
            # Cross-view cross-attention
            x0_new = layer["cross0"](x0, x1, x1)
            x1_new = layer["cross1"](x1, x0, x0)
            x0, x1 = x0_new, x1_new
        return x0, x1


# ==============================================================================
# 4. Log-Sinkhorn Optimal Transport with Dustbin
# ==============================================================================
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Iterative row/column log-sum-exp normalization."""
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=-1)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


class ObjectSinkhornMatcher(nn.Module):
    """
    Computes pairwise affinity and applies optimal transport with dustbins.
    Extends (M0, M1) matrix to (M0 + 1, M1 + 1) where index M0 (last row) and
    M1 (last col) are the dustbins for unmatched objects.
    """

    def __init__(self, d_model: int = 128, sinkhorn_iters: int = 20) -> None:
        super().__init__()
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters
        self.dustbin_score = nn.Parameter(torch.tensor(0.0))
        self.proj0 = nn.Linear(d_model, d_model, bias=False)
        self.proj1 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x0: (B, M0, d_model)
        x1: (B, M1, d_model)
        Returns:
          log_P: (B, M0 + 1, M1 + 1)
          P: (B, M0 + 1, M1 + 1) probability matrix
        """
        B, M0, _ = x0.shape
        _, M1, _ = x1.shape
        device = x0.device

        # Unnormalized projected similarity scaled by 1/sqrt(d_model)
        q0 = self.proj0(x0)
        k1 = self.proj1(x1)
        scores = torch.bmm(q0, k1.transpose(1, 2)) / math.sqrt(self.d_model)  # (B, M0, M1)

        # Augment with dustbins
        bin_val = self.dustbin_score.repeat(B, 1, 1)  # (B, 1, 1)
        bin_row = self.dustbin_score.repeat(B, 1, M1) # (B, 1, M1)
        bin_col = self.dustbin_score.repeat(B, M0, 1) # (B, M0, 1)

        top = torch.cat([scores, bin_col], dim=-1)       # (B, M0, M1 + 1)
        bottom = torch.cat([bin_row, bin_val], dim=-1)   # (B, 1, M1 + 1)
        Z = torch.cat([top, bottom], dim=1)              # (B, M0 + 1, M1 + 1)

        # SuperGlue Marginals: each object row/column must sum to 1.0
        # Dustbin row sums to M1 (total items in view 1)
        # Dustbin col sums to M0 (total items in view 0)
        log_mu = torch.cat([
            torch.zeros(M0, device=device),
            torch.tensor([math.log(float(M1) + 1e-6)], device=device),
        ]).repeat(B, 1)
        log_nu = torch.cat([
            torch.zeros(M1, device=device),
            torch.tensor([math.log(float(M0) + 1e-6)], device=device),
        ]).repeat(B, 1)

        log_P = log_sinkhorn_iterations(Z, log_mu, log_nu, iters=self.sinkhorn_iters)
        P = torch.exp(log_P)
        return log_P, P


# ==============================================================================
# 5. Full XFeatObjectMatcher Network & Loss
# ==============================================================================
class XFeatObjectMatcher(nn.Module):
    """
    End-to-end Object-Level Bipartite Matcher based on XFeat backbone.
    Inputs:
      img0: (B, 3, H, W)
      img1: (B, 3, H, W)
      masks0: (B, M0, H, W)
      bboxes0: (B, M0, 4) [x1, y1, x2, y2]
      masks1: (B, M1, H, W)
      bboxes1: (B, M1, 4) [x1, y1, x2, y2]
    Output dict with:
      'log_P': (B, M0 + 1, M1 + 1)
      'prob_matrix': (B, M0 + 1, M1 + 1)
      'tokens0': (B, M0, d_model)
      'tokens1': (B, M1, d_model)
    """

    def __init__(
        self,
        feat_dim: int = 64,
        d_model: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        sinkhorn_iters: int = 20,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # 1. XFeat CNN Backbone (shared weights across views)
        self.backbone = XFeatBackbone(in_channels=3, out_dim=feat_dim)

        # 2. Mask Pooling
        self.mask_pooling = MaskPooling()

        # 3. Spatial Geometry Encoder
        self.geo_encoder = ObjectGeometryEncoder(out_dim=feat_dim)

        # 4. Token Fusion Layer
        self.token_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

        # 5. Graph Transformer (Intra-view + Cross-view)
        self.transformer = ObjectGraphTransformer(d_model=d_model, num_layers=num_layers, nhead=nhead)

        # 6. Optimal Transport Head
        self.matcher = ObjectSinkhornMatcher(d_model=d_model, sinkhorn_iters=sinkhorn_iters)

    def extract_object_tokens(
        self, img: torch.Tensor, masks: torch.Tensor, bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Extracts fused visual+geometric tokens for a set of objects."""
        B, _, H, W = img.shape
        feat_map = self.backbone(img)                                # (B, feat_dim, H/8, W/8)
        v_feat = self.mask_pooling(feat_map, masks)                  # (B, M, feat_dim)
        g_feat = self.geo_encoder(bboxes, (H, W))                    # (B, M, feat_dim)
        tokens = self.token_proj(torch.cat([v_feat, g_feat], dim=-1)) # (B, M, d_model)
        return tokens

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        masks0: torch.Tensor,
        bboxes0: torch.Tensor,
        masks1: torch.Tensor,
        bboxes1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # 1. Extract object tokens for both views
        x0 = self.extract_object_tokens(img0, masks0, bboxes0)
        x1 = self.extract_object_tokens(img1, masks1, bboxes1)

        # 2. Context propagation via graph attention
        x0_ctx, x1_ctx = self.transformer(x0, x1)

        # 3. Sinkhorn matching with dustbin
        log_P, P = self.matcher(x0_ctx, x1_ctx)

        return {
            "log_P": log_P,
            "prob_matrix": P,
            "tokens0": x0_ctx,
            "tokens1": x1_ctx,
        }


class ObjectMatchingLoss(nn.Module):
    """
    SuperGlue-style Negative Log-Likelihood (NLL) Loss over Sinkhorn probability matrix.
    Supervises both:
      1. Co-visible ground-truth pairs: -log P(i, j)
      2. Out-of-view objects in View 0: -log P(i, M1) (View 0 dustbin)
      3. Out-of-view objects in View 1: -log P(M0, j) (View 1 dustbin)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        log_P: torch.Tensor,
        gt_matches: List[List[Tuple[int, int]]],
        unmatched_0: List[List[int]],
        unmatched_1: List[List[int]],
    ) -> torch.Tensor:
        """
        log_P: (B, M0 + 1, M1 + 1)
        gt_matches: list of batch elements, each is a list of (idx0, idx1)
        unmatched_0: list of batch elements, each is a list of idx0
        unmatched_1: list of batch elements, each is a list of idx1
        """
        B, M0_plus_1, M1_plus_1 = log_P.shape
        M0 = M0_plus_1 - 1
        M1 = M1_plus_1 - 1

        total_loss = torch.tensor(0.0, device=log_P.device)
        total_items = 0

        for b in range(B):
            matches = gt_matches[b]
            un0 = unmatched_0[b]
            un1 = unmatched_1[b]

            loss_b = torch.tensor(0.0, device=log_P.device)
            # 1. Matched pairs
            for i, j in matches:
                loss_b = loss_b - log_P[b, i, j]
                total_items += 1

            # 2. Unmatched in View 0 (dustbin column index M1)
            for i in un0:
                loss_b = loss_b - log_P[b, i, M1]
                total_items += 1

            # 3. Unmatched in View 1 (dustbin row index M0)
            for j in un1:
                loss_b = loss_b - log_P[b, M0, j]
                total_items += 1

            total_loss = total_loss + loss_b

        if total_items > 0:
            return total_loss / float(total_items)
        return total_loss
