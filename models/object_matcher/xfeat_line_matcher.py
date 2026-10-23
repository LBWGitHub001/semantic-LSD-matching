"""
XFeatLineMatcher: Multi-Modal Object & Straight Line Hybrid Bipartite Matcher ("连连看").

Architecture:
  1. Patch Extractor & Line Associator: Crops object visual patches and localizes straight lines in patch-relative space.
  2. Visual Patch Encoder: Multi-scale XFeat pyramid backbone (P1+P2+P3) extracting fine texture & semantic tokens.
  3. Straight Line Geometry Encoder: DeepSets-style permutation-invariant encoder for patch-internal LSD lines with null token fallback.
  4. Multi-Modal Fusion: Fuses visual token, line token, and global bounding-box layout into unified object-line token.
  5. Object-Line Graph Transformer: Cross-view attention exchanging contextual features between viewpoints.
  6. Optimal Transport Head: Log-Sinkhorn with learnable dustbin slack rejection for out-of-view / occluded objects.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.object_matcher.loss import ObjectMatchingLoss
from models.object_matcher.xfeat_object_matcher import (
    BasicResidualBlock,
    ObjectGeometryEncoder,
    ObjectGraphTransformer,
    ObjectSinkhornMatcher,
    XFeatBackbone,
    XFeatPyramidBackbone,
)


# ==============================================================================
# 1. Straight Line Geometry Encoder for Local Patches (局部小图直线几何编码器)
# ==============================================================================
class PatchLineGeometryEncoder(nn.Module):
    """
    Encodes variable numbers of straight line segments inside an object patch.

    Lines can be passed either as:
      - 5D descriptors: [c_u, c_v, length, sin(2θ), cos(2θ)] relative to patch coords [0, 1].
      - 4D endpoints: [u1, v1, u2, v2] in patch normalized coordinates [0, 1].

    Features:
      1. Trainable `null_line_token` (embedding) for patches containing 0 straight lines.
      2. 2θ orientation encoding avoiding 180-degree endpoint ambiguity.
      3. Permutation-invariant aggregation (DeepSets Max-Pool + Avg-Pool) over K lines.
    """

    def __init__(self, line_dim: int = 64) -> None:
        super().__init__()
        self.line_dim = line_dim

        # 5-dim line input -> line_dim embedding
        self.line_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, line_dim),
            nn.LayerNorm(line_dim),
        )

        # Multi-line aggregation projection: [max_pool || avg_pool] (2 * line_dim -> line_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(2 * line_dim, line_dim),
            nn.LayerNorm(line_dim),
            nn.ReLU(inplace=True),
        )

        # Learnable fallback embedding for objects without any detected straight lines
        self.null_line_token = nn.Parameter(torch.zeros(line_dim))
        nn.init.normal_(self.null_line_token, std=0.02)

    @staticmethod
    def endpoints_to_5d(lines_4d: torch.Tensor) -> torch.Tensor:
        """Converts [u1, v1, u2, v2] patch endpoints into 5D descriptors."""
        u1 = lines_4d[..., 0]
        v1 = lines_4d[..., 1]
        u2 = lines_4d[..., 2]
        v2 = lines_4d[..., 3]

        cu = (u1 + u2) / 2.0
        cv = (v1 + v2) / 2.0
        du = u2 - u1
        dv = v2 - v1
        length = torch.hypot(du, dv)
        theta = torch.atan2(dv, du)
        sin2 = torch.sin(2.0 * theta)
        cos2 = torch.cos(2.0 * theta)

        return torch.stack([cu, cv, length, sin2, cos2], dim=-1)

    def forward(
        self,
        lines: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
    ) -> torch.Tensor:
        """
        lines: (B, M, K, 5) or (B, M, K, 4) tensor of line segments.
        valid_mask: (B, M, K) boolean tensor where True indicates a real line.
        num_patches: (B, M) count if lines is None.

        Returns: (B, M, line_dim) straight line token per patch.
        """
        if lines is None or lines.numel() == 0:
            if num_patches is None:
                raise ValueError("Must provide either lines tensor or num_patches")
            B, M = num_patches
            return self.null_line_token.view(1, 1, self.line_dim).expand(B, M, self.line_dim)

        B, M, K, feat_dim = lines.shape
        if feat_dim == 4:
            geom_5d = self.endpoints_to_5d(lines)
        else:
            geom_5d = lines

        if valid_mask is None:
            # If line length > 1e-4 or features are non-zero, treat as valid
            valid_mask = (geom_5d[..., 2].abs() > 1e-4) | (geom_5d.abs().sum(dim=-1) > 1e-4)

        # Flatten B and M for batched MLP processing: (B*M, K, 5)
        flat_geom = geom_5d.view(B * M, K, 5)
        flat_mask = valid_mask.view(B * M, K)

        # Step 1: Encode individual lines: (B*M, K, line_dim)
        h = self.line_mlp(flat_geom)

        # Step 2: Masked Pooling over K lines
        mask_expanded = flat_mask.unsqueeze(-1)  # (B*M, K, 1)

        # FP16-safe max pooling (use -1e4 instead of -1e9 to stay within FP16 dynamic range [-65504, 65504])
        h_for_max = h.masked_fill(~mask_expanded, -1e4)
        max_pool = torch.max(h_for_max, dim=1)[0]  # (B*M, line_dim)

        # Average pooling
        h_for_avg = h.masked_fill(~mask_expanded, 0.0)
        counts = flat_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        avg_pool = h_for_avg.sum(dim=1) / counts  # (B*M, line_dim)

        pooled = self.pool_proj(torch.cat([max_pool, avg_pool], dim=-1))  # (B*M, line_dim)

        # Step 3: Handle patches where valid line count is 0 (fallback to null_line_token)
        has_lines = flat_mask.any(dim=-1, keepdim=True)  # (B*M, 1)
        null_expanded = self.null_line_token.view(1, self.line_dim).expand(B * M, self.line_dim)
        out = torch.where(has_lines, pooled, null_expanded)

        return out.view(B, M, self.line_dim)


# ==============================================================================
# 2. Region Patch Extractor & Line Associator Helper (全图小图与直线裁剪关联)
# ==============================================================================
class RegionPatchExtractor:
    """
    Extracts small object patches (e.g. 112x112) from full images and associates
    global straight lines with corresponding object patches.
    """

    @staticmethod
    def associate_lines_with_bboxes(
        lines: torch.Tensor,
        bboxes: torch.Tensor,
        max_lines_per_patch: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        lines: (B, N, 4) in image coordinates [x1, y1, x2, y2].
        bboxes: (B, M, 4) in image coordinates [x1, y1, x2, y2].

        Returns:
          patch_lines: (B, M, max_lines_per_patch, 5) normalized in [0, 1] relative to patch.
          valid_mask: (B, M, max_lines_per_patch) boolean tensor.
        """
        B, N, _ = lines.shape
        _, M, _ = bboxes.shape
        device = lines.device

        patch_lines = torch.zeros((B, M, max_lines_per_patch, 5), dtype=torch.float32, device=device)
        valid_mask = torch.zeros((B, M, max_lines_per_patch), dtype=torch.bool, device=device)

        if N == 0 or M == 0:
            return patch_lines, valid_mask

        for b in range(B):
            b_lines = lines[b]  # (N, 4)
            b_bboxes = bboxes[b]  # (M, 4)

            # Compute line midpoints
            mid_x = (b_lines[:, 0] + b_lines[:, 2]) / 2.0  # (N,)
            mid_y = (b_lines[:, 1] + b_lines[:, 3]) / 2.0  # (N,)

            for m in range(M):
                bx1, by1, bx2, by2 = b_bboxes[m]
                bw = torch.clamp(bx2 - bx1, min=1.0)
                bh = torch.clamp(by2 - by1, min=1.0)

                # Check which line midpoints fall inside bounding box (with slight margin)
                in_box = (
                    (mid_x >= bx1 - 0.05 * bw)
                    & (mid_x <= bx2 + 0.05 * bw)
                    & (mid_y >= by1 - 0.05 * bh)
                    & (mid_y <= by2 + 0.05 * bh)
                )

                matched_indices = torch.nonzero(in_box).flatten()
                if matched_indices.numel() == 0:
                    continue

                # Take top max_lines_per_patch lines
                sel_indices = matched_indices[:max_lines_per_patch]
                sel_lines = b_lines[sel_indices]  # (K, 4)
                K = sel_lines.shape[0]

                # Normalize line coordinates to patch space [0, 1]
                u1 = (sel_lines[:, 0] - bx1) / bw
                v1 = (sel_lines[:, 1] - by1) / bh
                u2 = (sel_lines[:, 2] - bx1) / bw
                v2 = (sel_lines[:, 3] - by1) / bh

                cu = (u1 + u2) / 2.0
                cv = (v1 + v2) / 2.0
                du = u2 - u1
                dv = v2 - v1
                l_norm = torch.hypot(du, dv)
                theta = torch.atan2(dv, du)
                sin2 = torch.sin(2.0 * theta)
                cos2 = torch.cos(2.0 * theta)

                geom_5d = torch.stack([cu, cv, l_norm, sin2, cos2], dim=-1)
                patch_lines[b, m, :K] = geom_5d
                valid_mask[b, m, :K] = True

        return patch_lines, valid_mask

    @staticmethod
    def crop_patches(
        images: torch.Tensor,
        bboxes: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        patch_size: int = 112,
    ) -> torch.Tensor:
        """
        Extracts cropped and resized patches from images using bboxes and optional masks.
        images: (B, 3, H, W)
        bboxes: (B, M, 4) in image coordinates [x1, y1, x2, y2]
        masks: (B, M, H, W) optional instance masks to isolate object from background.
        Returns: (B, M, 3, patch_size, patch_size)
        """
        B, C, H, W = images.shape
        _, M, _ = bboxes.shape
        device = images.device

        if M == 0:
            return torch.zeros((B, 0, C, patch_size, patch_size), dtype=images.dtype, device=device)

        patches = torch.zeros((B, M, C, patch_size, patch_size), dtype=images.dtype, device=device)

        for b in range(B):
            img = images[b]  # (3, H, W)
            for m in range(M):
                bx1 = int(torch.clamp(bboxes[b, m, 0], 0, W - 1).item())
                by1 = int(torch.clamp(bboxes[b, m, 1], 0, H - 1).item())
                bx2 = int(torch.clamp(bboxes[b, m, 2], bx1 + 1, W).item())
                by2 = int(torch.clamp(bboxes[b, m, 3], by1 + 1, H).item())

                crop = img[:, by1:by2, bx1:bx2]
                if masks is not None:
                    m_crop = masks[b, m, by1:by2, bx1:bx2].unsqueeze(0)
                    crop = crop * (m_crop > 0.5).float()

                # Bilinear resize to (patch_size, patch_size)
                crop_resized = F.interpolate(
                    crop.unsqueeze(0),
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                patches[b, m] = crop_resized

        return patches


# ==============================================================================
# 3. Main XFeatLineMatcher Network (多模态物体-直线融合连连看网络)
# ==============================================================================
class XFeatLineMatcher(nn.Module):
    """
    XFeatLineMatcher: Multi-Modal Bipartite Matching Network combining
    object visual patches and straight line geometry.

    Forward Pipeline:
      1. Receives either:
         (A) Whole images (B, 3, H, W), masks (B, M, H, W), and lines (B, N, 4).
         (B) Pre-cropped patches (B, M, 3, 112, 112) and patch lines (B, M, K, 5).
      2. Extracts visual token t_vis (192d) via XFeatPyramidBackbone.
      3. Extracts line token t_line (64d) via PatchLineGeometryEncoder.
      4. Extracts global layout token t_bbox (64d) via ObjectGeometryEncoder.
      5. Fuses tokens into unified t_fused (128d).
      6. Runs ObjectGraphTransformer cross-view contextual reasoning.
      7. Computes optimal transport matching via Log-Sinkhorn with dustbin.
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
        patch_size: int = 112,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.line_dim = line_dim
        self.d_model = d_model
        self.use_pyramid = use_pyramid
        self.with_geometry = with_geometry
        self.patch_size = patch_size

        # 1. Visual Backbone (Identical to XFeatObjectMatcher for maximum reuse)
        if self.use_pyramid:
            self.visual_backbone = XFeatPyramidBackbone(in_channels=3, feat_dim=feat_dim)
            self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
            vis_in_dim = 3 * feat_dim  # 192d
        else:
            self.visual_backbone = XFeatBackbone(in_channels=3, out_dim=feat_dim)
            self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
            vis_in_dim = feat_dim  # 64d

        # 2. Straight Line Geometry Encoder
        self.line_encoder = PatchLineGeometryEncoder(line_dim=line_dim)

        # 3. Global 2D Layout Geometry Encoder
        if self.with_geometry:
            self.geo_encoder = ObjectGeometryEncoder(out_dim=feat_dim)
            geo_dim = feat_dim
        else:
            self.geo_encoder = None
            geo_dim = 0

        # 4. Multi-Modal Token Fusion Layer: [t_vis || t_line || t_bbox] -> d_model
        total_fusion_dim = vis_in_dim + line_dim + geo_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(total_fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

        # 5. Graph Transformer & Log-Sinkhorn Matcher (100% reused from ObjectMatcher)
        self.transformer = ObjectGraphTransformer(d_model=d_model, num_layers=num_layers, nhead=nhead)
        self.matcher = ObjectSinkhornMatcher(d_model=d_model, sinkhorn_iters=sinkhorn_iters)

    def extract_visual_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, M, 3, H_p, W_p)
        Returns: (B, M, vis_in_dim)
        """
        B, M, C, H_p, W_p = patches.shape
        if M == 0:
            dim = 3 * self.feat_dim if self.use_pyramid else self.feat_dim
            return torch.zeros((B, 0, dim), device=patches.device, dtype=patches.dtype)

        flat_patches = patches.view(B * M, C, H_p, W_p)
        if self.use_pyramid:
            p1, p2, p3 = self.visual_backbone(flat_patches)
            f1 = self.visual_pool(p1).flatten(1)  # Stride 2 fine details
            f2 = self.visual_pool(p2).flatten(1)  # Stride 4 part shapes
            f3 = self.visual_pool(p3).flatten(1)  # Stride 8 semantic category
            vis_feat = torch.cat([f1, f2, f3], dim=-1)  # (B*M, 192)
        else:
            feat_map = self.visual_backbone(flat_patches)
            vis_feat = self.visual_pool(feat_map).flatten(1)  # (B*M, 64)

        return vis_feat.view(B, M, -1)

    def extract_fused_tokens(
        self,
        patches: torch.Tensor,
        patch_lines: Optional[torch.Tensor] = None,
        line_mask: Optional[torch.Tensor] = None,
        bboxes: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (480, 640),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts and fuses visual tokens and line tokens for a single view.
        Returns:
          t_fused: (B, M, d_model)
          t_vis: (B, M, vis_in_dim)
          t_line: (B, M, line_dim)
        """
        B, M = patches.shape[:2]
        if M == 0:
            empty = torch.zeros((B, 0, self.d_model), device=patches.device, dtype=patches.dtype)
            return empty, empty, empty

        # 1. Visual Token
        t_vis = self.extract_visual_tokens(patches)  # (B, M, vis_dim)

        # 2. Straight Line Token
        if patch_lines is not None and patch_lines.numel() > 0:
            t_line = self.line_encoder(patch_lines, valid_mask=line_mask)
        else:
            t_line = self.line_encoder(lines=None, num_patches=(B, M)).to(patches.device)

        # 3. Global Geometry Token (Optional)
        concat_list = [t_vis, t_line]
        if self.with_geometry:
            if bboxes is not None and bboxes.numel() > 0:
                t_bbox = self.geo_encoder(bboxes, image_size=image_size)
            else:
                t_bbox = torch.zeros((B, M, self.feat_dim), device=patches.device, dtype=patches.dtype)
            concat_list.append(t_bbox)

        # 4. Multi-Modal Fusion Projection
        fused_raw = torch.cat(concat_list, dim=-1)
        t_fused = self.fusion_proj(fused_raw)

        return t_fused, t_vis, t_line

    def forward_patches(
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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward method from pre-cropped patches and localized straight lines.
        Optimal for high-throughput batch training.
        """
        x0, vis0, line0 = self.extract_fused_tokens(
            patches0,
            patch_lines=patch_lines0,
            line_mask=line_mask0,
            bboxes=bboxes0,
            image_size=image_size,
        )
        x1, vis1, line1 = self.extract_fused_tokens(
            patches1,
            patch_lines=patch_lines1,
            line_mask=line_mask1,
            bboxes=bboxes1,
            image_size=image_size,
        )

        # Cross-View Contextual Transformer
        x0_ctx, x1_ctx = self.transformer(x0, x1)

        # Optimal Transport with Dustbin
        log_P, P = self.matcher(x0_ctx, x1_ctx)

        return {
            "log_P": log_P,
            "prob_matrix": P,
            "tokens0": x0_ctx,
            "tokens1": x1_ctx,
            "vis_tokens0": vis0,
            "vis_tokens1": vis1,
            "line_tokens0": line0,
            "line_tokens1": line1,
        }

    def forward_full_images(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        bboxes0: torch.Tensor,
        bboxes1: torch.Tensor,
        lines0: Optional[torch.Tensor] = None,
        lines1: Optional[torch.Tensor] = None,
        masks0: Optional[torch.Tensor] = None,
        masks1: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-End Forward method: Takes full images, masks/bboxes, and global LSD lines.
        Automatically crops object patches and localizes lines inside patches.
        """
        H, W = img0.shape[-2:]

        # 1. Crop patches
        patches0 = RegionPatchExtractor.crop_patches(img0, bboxes0, masks=masks0, patch_size=self.patch_size)
        patches1 = RegionPatchExtractor.crop_patches(img1, bboxes1, masks=masks1, patch_size=self.patch_size)

        # 2. Localize lines within patches
        if lines0 is not None and lines0.numel() > 0:
            p_lines0, p_mask0 = RegionPatchExtractor.associate_lines_with_bboxes(lines0, bboxes0)
        else:
            p_lines0, p_mask0 = None, None

        if lines1 is not None and lines1.numel() > 0:
            p_lines1, p_mask1 = RegionPatchExtractor.associate_lines_with_bboxes(lines1, bboxes1)
        else:
            p_lines1, p_mask1 = None, None

        return self.forward_patches(
            patches0=patches0,
            patches1=patches1,
            patch_lines0=p_lines0,
            patch_lines1=p_lines1,
            line_mask0=p_mask0,
            line_mask1=p_mask1,
            bboxes0=bboxes0,
            bboxes1=bboxes1,
            image_size=(H, W),
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Adaptive polymorphic forward dispatcher:
        - If first arguments are 5D patches (B, M, 3, H, W) -> dispatches to `forward_patches`.
        - If first arguments are 4D full images (B, 3, H, W) -> dispatches to `forward_full_images`.
        """
        # Check keyword arguments first
        if "patches0" in kwargs:
            return self.forward_patches(**kwargs)
        if "img0" in kwargs or "images0" in kwargs:
            img0 = kwargs.pop("img0", kwargs.pop("images0", None))
            img1 = kwargs.pop("img1", kwargs.pop("images1", None))
            return self.forward_full_images(img0=img0, img1=img1, **kwargs)

        # Check positional arguments
        if len(args) >= 2:
            arg0, arg1 = args[0], args[1]
            if isinstance(arg0, torch.Tensor) and arg0.ndim == 5:
                # (B, M, 3, H, W) -> patches
                return self.forward_patches(*args, **kwargs)
            elif isinstance(arg0, torch.Tensor) and arg0.ndim == 4:
                # (B, 3, H, W) -> full images
                return self.forward_full_images(*args, **kwargs)

        raise ValueError("Invalid forward inputs: provide either patches or full images with bboxes/masks.")
