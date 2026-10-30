"""Descriptor hinge loss. The interest-point head is not used."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPointLoss(nn.Module):
    """描述子合页损失。

    单应把图 1 的格子中心变到图 2，距离不超过 8 像素算正样本。
    正样本再乘 250，因为负样本多得多。正间隔是 1，负间隔是 0.2。
    检测头去掉之后，这一项不再乘 0.0001。
    """

    def __init__(
        self,
        positive_weight: float = 250.0,
        positive_margin: float = 1.0,
        negative_margin: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.positive_weight = float(positive_weight)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)

    def forward(
        self,
        pred0: dict[str, torch.Tensor],
        pred1: dict[str, torch.Tensor],
        homography: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        descriptor = self._descriptor_loss(pred0["desc"], pred1["desc"], homography)
        return {"loss": descriptor, "ld": descriptor.detach()}

    def _descriptor_loss(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        homography: torch.Tensor,
    ) -> torch.Tensor:
        # desc 是 (N, 256, Hc, Wc)，已经按通道归一化。
        batch, _, cells_h, cells_w = desc0.shape
        flat0 = desc0.permute(0, 2, 3, 1).reshape(batch, -1, desc0.shape[1])
        flat1 = desc1.permute(0, 2, 3, 1).reshape(batch, -1, desc1.shape[1])
        dot = torch.matmul(flat0, flat1.transpose(1, 2))
        centers = _cell_centers(cells_h, cells_w, desc0.device, desc0.dtype)
        centers = centers.reshape(1, -1, 2).expand(batch, -1, -1)
        warped = _apply_homography(homography, centers)
        distance = torch.cdist(warped, centers)
        positive = (distance <= 8).to(dot.dtype)
        hinge_pos = self.positive_weight * positive * F.relu(self.positive_margin - dot)
        hinge_neg = (1.0 - positive) * F.relu(dot - self.negative_margin)
        return (hinge_pos + hinge_neg).mean()


def _cell_centers(cells_h: int, cells_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """每个格子中心在原图像素坐标里的位置。格子边长是 8。"""
    rows = (torch.arange(cells_h, device=device, dtype=dtype) + 0.5) * 8 - 0.5
    cols = (torch.arange(cells_w, device=device, dtype=dtype) + 0.5) * 8 - 0.5
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def _apply_homography(homography: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
    homo = torch.cat([points, ones], dim=-1)
    mapped = torch.einsum("bij,bmj->bmi", homography, homo)
    return mapped[..., :2] / mapped[..., 2:3].clamp(min=1e-6)
