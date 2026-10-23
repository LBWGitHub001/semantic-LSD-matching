"""
Object-Level Bipartite Matcher ("连连看") Models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.object_matcher.loss import ObjectMatchingLoss


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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class XFeatPyramidBackbone(nn.Module):
    """
    Multi-Scale Feature Pyramid Network (FPN) backbone inspired by XFeat (CVPR 2024).
    Produces a multi-scale feature pyramid (P1, P2, P3) across 3 distinct spatial resolutions:
      - P1 (Stride 2): fine-grained local textures, edges and boundary details.
      - P2 (Stride 4): part-level geometry and structural shapes.
      - P3 (Stride 8): high-level semantic category context.
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 64) -> None:
        super().__init__()
        self.feat_dim = feat_dim

        # Bottom-up stages
        # Stage 1: Stride 2 (H/2, W/2) -> 32 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            BasicResidualBlock(32),
        )
        # Stage 2: Stride 2 (H/4, W/4) -> 64 channels
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicResidualBlock(64),
        )
        # Stage 3: Stride 2 (H/8, W/8) -> 128 channels
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            BasicResidualBlock(128),
        )

        # Top-down lateral connections (1x1 convs)
        self.lat3 = nn.Conv2d(128, feat_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(64, feat_dim, kernel_size=1)
        self.lat1 = nn.Conv2d(32, feat_dim, kernel_size=1)

        # Anti-aliasing smooth convs (3x3 convs)
        self.smooth2 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)  # Stride 2, C=32
        c2 = self.stage2(c1) # Stride 4, C=64
        c3 = self.stage3(c2) # Stride 8, C=128

        p3 = self.lat3(c3)
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest"))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest"))

        return p1, p2, p3


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

        if (H_m, W_m) != (H_f, W_f):
            masks_down = F.interpolate(masks, size=(H_f, W_f), mode="nearest")
        else:
            masks_down = masks

        m_flat = masks_down.view(B, M, 1, H_f * W_f)
        f_flat = feat_map.view(B, 1, C, H_f * W_f)

        num = (f_flat * m_flat).sum(dim=-1)
        den = m_flat.sum(dim=-1) + self.eps

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
# 3. Object Graph Transformer
# ==============================================================================
class AttentionBlock(nn.Module):
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
    def __init__(self, d_model: int = 128, num_layers: int = 2, nhead: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "self0": AttentionBlock(d_model, nhead),
                "self1": AttentionBlock(d_model, nhead),
                "cross0": AttentionBlock(d_model, nhead),
                "cross1": AttentionBlock(d_model, nhead),
            }))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x0 = layer["self0"](x0, x0, x0)
            x1 = layer["self1"](x1, x1, x1)
            x0_new = layer["cross0"](x0, x1, x1)
            x1_new = layer["cross1"](x1, x0, x0)
            x0, x1 = x0_new, x1_new
        return x0, x1


# ==============================================================================
# 4. Log-Sinkhorn Optimal Transport with Dustbin
# ==============================================================================
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int = 20) -> torch.Tensor:
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=-1)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


class ObjectSinkhornMatcher(nn.Module):
    def __init__(self, d_model: int = 128, sinkhorn_iters: int = 20) -> None:
        super().__init__()
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters
        self.dustbin_score = nn.Parameter(torch.tensor(0.0))
        self.proj0 = nn.Linear(d_model, d_model, bias=False)
        self.proj1 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M0, _ = x0.shape
        _, M1, _ = x1.shape
        device = x0.device

        q0 = self.proj0(x0)
        k1 = self.proj1(x1)
        scores = torch.bmm(q0, k1.transpose(1, 2)) / math.sqrt(self.d_model)

        bin_val = self.dustbin_score.repeat(B, 1, 1)
        bin_row = self.dustbin_score.repeat(B, 1, M1)
        bin_col = self.dustbin_score.repeat(B, M0, 1)

        top = torch.cat([scores, bin_col], dim=-1)
        bottom = torch.cat([bin_row, bin_val], dim=-1)
        Z = torch.cat([top, bottom], dim=1)

        log_mu = torch.cat([
            torch.zeros(M0, device=device),
            torch.tensor([math.log(float(M1) + 1e-6)], device=device),
        ]).repeat(B, 1)
        log_nu = torch.cat([
            torch.zeros(M1, device=device),
            torch.tensor([math.log(float(M0) + 1e-6)], device=device),
        ]).repeat(B, 1)

        # Run Sinkhorn optimal transport in FP32 to prevent FP16 logsumexp overflow (>65504)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            log_P = log_sinkhorn_iterations(Z.float(), log_mu.float(), log_nu.float(), iters=self.sinkhorn_iters)
            # Safe clamping against numerical underflow
            log_P = torch.clamp(log_P, min=-100.0, max=0.0)
            P = torch.exp(log_P)
        return log_P, P


# ==============================================================================
# 5. Full XFeatObjectMatcher Model (Multi-Scale Region Patch Bipartite Matcher)
# ==============================================================================
class XFeatObjectMatcher(nn.Module):
    """
    Multi-Scale Region Patch Bipartite Matcher ("连连看").
    Takes two sets of small region image patches (B, M0, 3, H_p, W_p) and (B, M1, 3, H_p, W_p),
    extracts multi-scale feature pyramids (P1, P2, P3), pools and fuses them across scales,
    and outputs bipartite matching probabilities through graph transformer and Sinkhorn.
    Masks and full images DO NOT enter the network.

    Backward-compatible with:
      - Multi-scale FPN pyramid models (use_pyramid=True)
      - Single-scale lightweight models (use_pyramid=False)
      - Legacy layout geometry-encoder models (with_geometry=True)
    """

    def __init__(
        self,
        feat_dim: int = 64,
        d_model: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        sinkhorn_iters: int = 20,
        use_pyramid: bool = True,
        with_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.feat_dim = feat_dim
        self.use_pyramid = use_pyramid
        self.with_geometry = with_geometry

        if self.use_pyramid:
            self.backbone = XFeatPyramidBackbone(in_channels=3, feat_dim=feat_dim)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            in_dim = 3 * feat_dim + (feat_dim if self.with_geometry else 0)
        else:
            self.backbone = XFeatBackbone(in_channels=3, out_dim=feat_dim)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            in_dim = feat_dim + (feat_dim if self.with_geometry else 0)

        if self.with_geometry:
            self.geo_encoder = ObjectGeometryEncoder(out_dim=feat_dim)

        self.token_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

        self.transformer = ObjectGraphTransformer(d_model=d_model, num_layers=num_layers, nhead=nhead)
        self.matcher = ObjectSinkhornMatcher(d_model=d_model, sinkhorn_iters=sinkhorn_iters)

    def extract_patch_tokens(
        self,
        patches: torch.Tensor,
        bboxes: torch.Tensor | None = None,
        image_size: Tuple[int, int] = (480, 640),
    ) -> torch.Tensor:
        """
        patches: (B, M, 3, H_p, W_p), e.g. (B, M, 3, 112, 112)
        Returns: (B, M, d_model)
        """
        B, M, C, H_p, W_p = patches.shape
        if M == 0:
            return torch.zeros((B, 0, self.d_model), device=patches.device, dtype=patches.dtype)

        flat_patches = patches.view(B * M, C, H_p, W_p)
        if self.use_pyramid:
            p1, p2, p3 = self.backbone(flat_patches)
            f1 = self.pool(p1).flatten(1)  # (B*M, feat_dim) - Stride 2 fine texture / edge
            f2 = self.pool(p2).flatten(1)  # (B*M, feat_dim) - Stride 4 part / geometric shape
            f3 = self.pool(p3).flatten(1)  # (B*M, feat_dim) - Stride 8 macro semantic category
            pooled = torch.cat([f1, f2, f3], dim=-1)  # (B*M, 3 * feat_dim)
        else:
            feat_map = self.backbone(flat_patches)  # (B*M, feat_dim, H_p/8, W_p/8)
            pooled = self.pool(feat_map).flatten(1)  # (B*M, feat_dim)

        if self.with_geometry:
            if bboxes is not None and bboxes.numel() > 0:
                g_feat = self.geo_encoder(bboxes, image_size).view(B * M, self.feat_dim)
            else:
                g_feat = torch.zeros((B * M, self.feat_dim), device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, g_feat], dim=-1)

        tokens = self.token_proj(pooled).view(B, M, self.d_model)
        return tokens

    # Alias for backward compatibility
    extract_object_tokens = extract_patch_tokens

    def forward(
        self,
        patches0: torch.Tensor,
        patches1: torch.Tensor,
        bboxes0: torch.Tensor | None = None,
        bboxes1: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Directly matches two collections of small region image patches.
        patches0: (B, M0, 3, 112, 112)
        patches1: (B, M1, 3, 112, 112)
        """
        x0 = self.extract_patch_tokens(patches0, bboxes=bboxes0)
        x1 = self.extract_patch_tokens(patches1, bboxes=bboxes1)
        x0_ctx, x1_ctx = self.transformer(x0, x1)
        log_P, P = self.matcher(x0_ctx, x1_ctx)
        return {
            "log_P": log_P,
            "prob_matrix": P,
            "tokens0": x0_ctx,
            "tokens1": x1_ctx,
        }
