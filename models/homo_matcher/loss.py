"""Cycle loss on the two homographies, plus depth profiles along ridges."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_homography(homography: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Map ``(B, ..., 2)`` pixel points with a batch of 3x3 homographies."""
    ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
    homo = torch.cat([points, ones], dim=-1)
    mapped = torch.einsum("bij,b...j->b...i", homography, homo)
    return mapped[..., :2] / mapped[..., 2:3].clamp(min=1e-6)


def _sample_depth(depth: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``depth`` is (B, 1, H, W). ``points`` is (B, L, N, 2) in pixels."""
    height, width = depth.shape[-2:]
    x = points[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = points[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1)
    batch, lines, count, _ = grid.shape
    sampled = F.grid_sample(
        depth,
        grid.view(batch, lines * count, 1, 2),
        align_corners=True,
        padding_mode="zeros",
    )
    return sampled.view(batch, lines, count)


def _zscore(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(values.dtype)
    count = mask.sum(dim=-1).clamp(min=1.0)
    mean = (values * mask).sum(dim=-1) / count
    centered = values - mean.unsqueeze(-1)
    variance = ((centered * centered) * mask).sum(dim=-1) / count
    return centered / variance.sqrt().clamp(min=1e-3).unsqueeze(-1)


class HomoMatcherLoss(nn.Module):
    """L = λ1 L1 + λ2 L2.

    L1 is the sum of squares of H12 H21 - I and H21 H12 - I.
    L2 compares z-scored depth samples along a ridge and along its image
    under H12. Points that leave the image or hit invalid depth are dropped.
    """

    def __init__(self, lambda1: float = 1.0, lambda2: float = 1.0, min_valid: int = 4, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.min_valid = int(min_valid)

    def forward(self, pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        homography_12 = pred["H12"].float()
        homography_21 = pred["H21"].float()
        identity = torch.eye(3, device=homography_12.device, dtype=homography_12.dtype).expand_as(homography_12)
        cycle = homography_12 @ homography_21 - identity
        cycle_back = homography_21 @ homography_12 - identity
        loss_cycle = (cycle.square().sum(dim=(1, 2)) + cycle_back.square().sum(dim=(1, 2))).mean()
        loss_depth = self._depth_loss(homography_12, batch)
        total = self.lambda1 * loss_cycle + self.lambda2 * loss_depth
        return {"loss": total, "l1": loss_cycle.detach(), "l2": loss_depth.detach()}

    def _depth_loss(self, homography_12: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ridges = batch["ridges0"].float()
        valid_lines = batch["ridge_mask"].bool()
        depth0 = batch["depth0"].float()
        depth1 = batch["depth1"].float()
        if depth0.dim() == 3:
            depth0 = depth0.unsqueeze(1)
            depth1 = depth1.unsqueeze(1)
        height, width = depth0.shape[-2:]
        samples0 = _sample_depth(depth0, ridges)
        warped = apply_homography(homography_12, ridges)
        samples1 = _sample_depth(depth1, warped)
        inside = (
            (warped[..., 0] >= 0)
            & (warped[..., 0] <= (width - 1))
            & (warped[..., 1] >= 0)
            & (warped[..., 1] <= (height - 1))
        )
        valid = inside & (samples0 > 1e-4) & (samples1 > 1e-4) & valid_lines.unsqueeze(-1)
        enough = valid.sum(dim=-1) >= self.min_valid
        if not bool(enough.any()):
            return homography_12.sum() * 0.0
        profile0 = _zscore(samples0, valid)
        profile1 = _zscore(samples1, valid)
        per_line = ((profile0 - profile1).square() * valid.to(profile0.dtype)).sum(dim=-1)
        per_line = per_line / valid.sum(dim=-1).clamp(min=1).to(per_line.dtype)
        per_line = per_line * enough.to(per_line.dtype)
        return per_line.sum() / enough.sum().clamp(min=1).to(per_line.dtype)
