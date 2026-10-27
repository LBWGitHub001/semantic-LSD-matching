"""
SemanticLineMatcher: dual region + line Sinkhorn matching.

Region branch reuses XFeat backbone / graph transformer / ObjectSinkhornMatcher.
Line branch (step06): along-line texture || region feature → line Sinkhorn.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.object_matcher.xfeat_object_matcher import (
    ObjectGeometryEncoder,
    ObjectGraphTransformer,
    ObjectSinkhornMatcher,
    XFeatBackbone,
    XFeatPyramidBackbone,
)
from models.semantic_line_matcher.line_encoder import AlongLineTextureEncoder


class SemanticLineMatcher(nn.Module):
    """
    Dual-output matcher under semantic regions.

    Returns region log_P and (from step06) line log_P_line with parent indices.
    """

    def __init__(
        self,
        feat_dim: int = 64,
        line_dim: int = 64,
        d_model: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        sinkhorn_iters: int = 20,
        use_pyramid: bool = True,
        with_geometry: bool = True,
        line_sample_points: int = 8,
        enable_line_head: bool = False,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.line_dim = line_dim
        self.d_model = d_model
        self.use_pyramid = use_pyramid
        self.with_geometry = with_geometry
        self.enable_line_head = bool(enable_line_head)
        self.line_sample_points = int(line_sample_points)

        if self.use_pyramid:
            self.visual_backbone = XFeatPyramidBackbone(in_channels=3, feat_dim=feat_dim)
            self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
            vis_in_dim = 3 * feat_dim
            # Prefer higher-res P2 for along-line sampling
            self.line_feat_channels = feat_dim
        else:
            self.visual_backbone = XFeatBackbone(in_channels=3, out_dim=feat_dim)
            self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
            vis_in_dim = feat_dim
            self.line_feat_channels = feat_dim

        if self.with_geometry:
            self.geo_encoder = ObjectGeometryEncoder(out_dim=feat_dim)
            geo_dim = feat_dim
        else:
            self.geo_encoder = None
            geo_dim = 0

        # Region token fusion: visual (+ optional geometry). Line pool not required for region.
        self.region_fusion = nn.Sequential(
            nn.Linear(vis_in_dim + geo_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.transformer = ObjectGraphTransformer(d_model=d_model, num_layers=num_layers, nhead=nhead)
        self.region_matcher = ObjectSinkhornMatcher(d_model=d_model, sinkhorn_iters=sinkhorn_iters)

        # Line head modules created always; used when enable_line_head=True (step06).
        self.line_encoder = AlongLineTextureEncoder(
            in_channels=self.line_feat_channels,
            out_dim=line_dim,
            num_samples=line_sample_points,
        )
        self.line_desc_proj = nn.Sequential(
            nn.Linear(line_dim + vis_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.line_matcher = ObjectSinkhornMatcher(d_model=d_model, sinkhorn_iters=sinkhorn_iters)
        self.vis_in_dim = vis_in_dim

    def _encode_patches(
        self,
        patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        patches: (B, M, 3, H, W)
        Returns:
          f_r: (B, M, vis_in_dim)
          fmap_line: (B*M, C, Hf, Wf) feature map for along-line sampling
        """
        B, M, C, H, W = patches.shape
        if M == 0:
            empty = torch.zeros((B, 0, self.vis_in_dim), device=patches.device, dtype=patches.dtype)
            dummy_map = torch.zeros((0, self.line_feat_channels, 1, 1), device=patches.device, dtype=patches.dtype)
            return empty, dummy_map

        flat = patches.view(B * M, C, H, W)
        if self.use_pyramid:
            p1, p2, p3 = self.visual_backbone(flat)
            f1 = self.visual_pool(p1).flatten(1)
            f2 = self.visual_pool(p2).flatten(1)
            f3 = self.visual_pool(p3).flatten(1)
            f_r = torch.cat([f1, f2, f3], dim=-1).view(B, M, -1)
            fmap_line = p2  # stride-4 texture
        else:
            feat_map = self.visual_backbone(flat)
            f_r = self.visual_pool(feat_map).flatten(1).view(B, M, -1)
            fmap_line = feat_map
        return f_r, fmap_line

    def _region_tokens(
        self,
        f_r: torch.Tensor,
        bboxes: Optional[torch.Tensor],
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        B, M, _ = f_r.shape
        parts = [f_r]
        if self.with_geometry:
            if bboxes is not None and bboxes.numel() > 0 and M > 0:
                t_geo = self.geo_encoder(bboxes, image_size=image_size)
            else:
                t_geo = torch.zeros((B, M, self.feat_dim), device=f_r.device, dtype=f_r.dtype)
            parts.append(t_geo)
        return self.region_fusion(torch.cat(parts, dim=-1))

    def forward_region(
        self,
        patches0: torch.Tensor,
        patches1: torch.Tensor,
        bboxes0: Optional[torch.Tensor] = None,
        bboxes1: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (480, 640),
    ) -> Dict[str, torch.Tensor]:
        f_r0, fmap0 = self._encode_patches(patches0)
        f_r1, fmap1 = self._encode_patches(patches1)
        x0 = self._region_tokens(f_r0, bboxes0, image_size)
        x1 = self._region_tokens(f_r1, bboxes1, image_size)
        x0, x1 = self.transformer(x0, x1)
        log_P, P = self.region_matcher(x0, x1)
        return {
            "log_P": log_P,
            "P": P,
            "prob_matrix": P,
            "tokens0": x0,
            "tokens1": x1,
            "f_r0": f_r0,
            "f_r1": f_r1,
            "fmap0": fmap0,
            "fmap1": fmap1,
        }

    def forward(
        self,
        patches0: torch.Tensor,
        patches1: torch.Tensor,
        patch_lines0: Optional[torch.Tensor] = None,
        patch_lines1: Optional[torch.Tensor] = None,
        line_mask0: Optional[torch.Tensor] = None,
        line_mask1: Optional[torch.Tensor] = None,
        bboxes0: Optional[torch.Tensor] = None,
        bboxes1: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (480, 640),
        **kwargs: Any,
    ) -> Dict[str, Any]:
        out = self.forward_region(
            patches0,
            patches1,
            bboxes0=bboxes0,
            bboxes1=bboxes1,
            image_size=image_size,
        )
        # Line head filled in step06
        if not self.enable_line_head:
            B = patches0.shape[0]
            device = patches0.device
            out["log_P_line"] = None
            out["P_line"] = None
            out["line_parents0"] = [torch.zeros(0, dtype=torch.long, device=device) for _ in range(B)]
            out["line_parents1"] = [torch.zeros(0, dtype=torch.long, device=device) for _ in range(B)]
        return out
