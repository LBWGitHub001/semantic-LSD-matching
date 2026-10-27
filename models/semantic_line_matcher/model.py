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
        enable_line_head: bool = True,
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

    def _line_descriptors_for_view(
        self,
        fmap: torch.Tensor,
        f_r: torch.Tensor,
        patch_lines: Optional[torch.Tensor],
        line_mask: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        fmap: (B*M, C, H, W), f_r: (B, M, Cv)
        Returns lists length B of (N_b, d_model) descriptors and (N_b,) parent region ids.
        """
        B, M, Cv = f_r.shape
        device = f_r.device
        dtype = f_r.dtype
        desc_list: List[torch.Tensor] = []
        parent_list: List[torch.Tensor] = []

        if M == 0 or patch_lines is None or patch_lines.numel() == 0:
            for _ in range(B):
                desc_list.append(torch.zeros((0, self.d_model), device=device, dtype=dtype))
                parent_list.append(torch.zeros((0,), device=device, dtype=torch.long))
            return desc_list, parent_list

        K = patch_lines.shape[2]
        if line_mask is None:
            line_mask = torch.ones((B, M, K), device=device, dtype=torch.bool)
        # Along-line encode per region instance
        lines_flat = patch_lines.view(B * M, K, -1)
        mask_flat = line_mask.view(B * M, K)
        f_line_all = self.line_encoder(fmap, lines_flat, valid_mask=mask_flat)  # (B*M, K, line_dim)
        f_line_all = f_line_all.view(B, M, K, self.line_dim)
        f_r_exp = f_r.unsqueeze(2).expand(-1, -1, K, -1)
        fused = torch.cat([f_line_all, f_r_exp], dim=-1)
        d_all = self.line_desc_proj(fused)  # (B, M, K, d_model)

        for b in range(B):
            descs = []
            parents = []
            for m in range(M):
                for k in range(K):
                    if not bool(line_mask[b, m, k].item()):
                        continue
                    descs.append(d_all[b, m, k])
                    parents.append(m)
            if descs:
                desc_list.append(torch.stack(descs, dim=0))
                parent_list.append(torch.tensor(parents, device=device, dtype=torch.long))
            else:
                desc_list.append(torch.zeros((0, self.d_model), device=device, dtype=dtype))
                parent_list.append(torch.zeros((0,), device=device, dtype=torch.long))
        return desc_list, parent_list

    def _match_lines_batch(
        self,
        desc0_list: List[torch.Tensor],
        desc1_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        log_list: List[torch.Tensor] = []
        p_list: List[torch.Tensor] = []
        for d0, d1 in zip(desc0_list, desc1_list):
            # ObjectSinkhornMatcher expects (B, M, D)
            log_P, P = self.line_matcher(d0.unsqueeze(0), d1.unsqueeze(0))
            log_list.append(log_P[0])
            p_list.append(P[0])
        return log_list, p_list

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
        B = patches0.shape[0]
        device = patches0.device

        if not self.enable_line_head:
            out["log_P_line"] = None
            out["P_line"] = None
            out["line_parents0"] = [torch.zeros(0, dtype=torch.long, device=device) for _ in range(B)]
            out["line_parents1"] = [torch.zeros(0, dtype=torch.long, device=device) for _ in range(B)]
            return out

        desc0, parents0 = self._line_descriptors_for_view(
            out["fmap0"], out["f_r0"], patch_lines0, line_mask0
        )
        desc1, parents1 = self._line_descriptors_for_view(
            out["fmap1"], out["f_r1"], patch_lines1, line_mask1
        )
        log_line, p_line = self._match_lines_batch(desc0, desc1)
        # For B==1 (training collate), also expose tensor forms
        out["log_P_line"] = log_line[0] if B == 1 else log_line
        out["P_line"] = p_line[0] if B == 1 else p_line
        out["log_P_line_list"] = log_line
        out["P_line_list"] = p_line
        out["line_parents0"] = parents0
        out["line_parents1"] = parents1
        out["line_desc0"] = desc0
        out["line_desc1"] = desc1
        return out

