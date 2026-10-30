"""Symmetry loss: the two velocity maps are inverses."""

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


class FCNFieldLoss(nn.Module):
    """只训练对称损失：T12 和 T21 互为逆映射。

    网络给出的是速度，不是矩阵。一个像素 x=(列, 行) 被速度挪到新位置：

        T12(x) = x + v12(x)     图1的点，按 v12 走到图2
        T21(x) = x + v21(x)     图2的点，按 v21 走回图1

    互逆就是走过去再走回来，还停在原来的像素上：
        T21(T12(x)) = x
        T12(T21(x)) = x
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        del kwargs

    def forward(self, pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 当前这项损失不用深度、内参或亮脊，batch 先空着，下一项损失再读。
        del batch
        velocity12 = pred["velocity12"].float()
        velocity21 = pred["velocity21"].float()
        symmetry = self.symmetry_loss(velocity12, velocity21)
        # symmetry 摘掉梯度，只给日志看；loss 还连着网络。
        return {"loss": symmetry, "symmetry": symmetry.detach()}

    def symmetry_loss(self, velocity12: torch.Tensor, velocity21: torch.Tensor) -> torch.Tensor:
        """两个方向各走一遍，取平均。velocity 的形状是 (B, 2, H, W)，每个像素都参与。"""
        _, _, height, width = velocity12.shape
        # 图1的每个像素先走 v12，再在落点读 v21。反过来再做一次。
        forward = self._one_cycle(velocity12, velocity21, height, width)
        backward = self._one_cycle(velocity21, velocity12, height, width)
        return 0.5 * (forward + backward)

    def _one_cycle(
        self,
        going: torch.Tensor,
        returning: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """整张图做一次往返。going、returning 都是 (B, 2, H, W)。

        出发速度就是 going 本身，和像素网格对齐，不用再采样。
        落点 = 像素坐标 + 出发速度，一般落在小数位置，回来的速度要在落点上插值读取。
        两段位移相加应为 0。落点跑出画面的像素丢掉。
        """
        rows = torch.arange(height, device=going.device, dtype=going.dtype)
        cols = torch.arange(width, device=going.device, dtype=going.dtype)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        landed_x = grid_x + going[:, 0]
        landed_y = grid_y + going[:, 1]
        back = _sample_field(returning, landed_x, landed_y)
        inside = (
            (landed_x >= 0)
            & (landed_x <= (width - 1))
            & (landed_y >= 0)
            & (landed_y <= (height - 1))
        )
        residual = (going + back).square().sum(dim=1) * inside.to(going.dtype)
        return residual.sum() / inside.to(going.dtype).sum().clamp(min=1)


def _sample_field(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """在速度图的每个像素位置上取值。x、y 是像素坐标，形状 (B, H, W)。

    grid_sample 要的坐标是 [-1, 1]。align_corners=True 时，像素 0 对应 -1，
    像素 width-1 对应 +1。返回形状与 field 相同。
    """
    _, _, height, width = field.shape
    x_norm = x / max(width - 1, 1) * 2.0 - 1.0
    y_norm = y / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(field, grid, align_corners=True, padding_mode="zeros")


def _sample_vector(field: torch.Tensor, points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """在速度图上按像素坐标取值。field 是 (B, 2, H, W)，points 是 (B, N, 2)。

    grid_sample 要的坐标是 [-1, 1]，不是像素。align_corners=True 时，
    像素 0 对应 -1，像素 width-1 对应 +1。返回 (B, N, 2)。
    """
    x = points[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = points[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(field, grid, align_corners=True)
    return sampled.squeeze(-1).transpose(1, 2).contiguous()


def warp_by_velocity(velocity: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Move pixel points by the velocity sampled at those points."""
    height, width = velocity.shape[-2:]
    flat = points.reshape(points.shape[0], -1, 2)
    moved = flat + _sample_vector(velocity, flat, height, width)
    return moved.reshape(points.shape)


def _masked_vector_l2(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(predicted.dtype)
    if valid.dim() == 4:
        mask = mask.expand_as(predicted)
    error = (predicted - target).square() * mask
    count = mask.sum().clamp(min=1)
    if not bool(valid.any()):
        return predicted.sum() * 0.0
    return error.sum() / count


def geometric_flow(
    depth_src: torch.Tensor,
    depth_dst: torch.Tensor,
    intrinsics_src: torch.Tensor,
    intrinsics_dst: torch.Tensor,
    w2c_src: torch.Tensor,
    w2c_dst: torch.Tensor,
    occlusion_m: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense displacement from depth and known cameras.

    Depth and the camera translation are in meters. Returns flow ``(B, 2, H, W)``
    as (du, dv) and a valid mask ``(B, 1, H, W)``.
    """
    if depth_src.dim() == 3:
        depth_src = depth_src.unsqueeze(1)
        depth_dst = depth_dst.unsqueeze(1)
    batch, _, height, width = depth_src.shape
    device = depth_src.device
    dtype = depth_src.dtype
    rows = torch.arange(height, device=device, dtype=dtype)
    cols = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    pixels = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
    pixels = pixels.unsqueeze(0).expand(batch, -1, -1, -1)
    rays = torch.einsum("bij,bjhw->bihw", torch.inverse(intrinsics_src), pixels)
    camera = rays * depth_src
    camera_to_world = torch.inverse(w2c_src)
    world = torch.einsum("bij,bjhw->bihw", camera_to_world[:, :3, :3], camera)
    world = world + camera_to_world[:, :3, 3].view(batch, 3, 1, 1)
    destination = torch.einsum("bij,bjhw->bihw", w2c_dst[:, :3, :3], world)
    destination = destination + w2c_dst[:, :3, 3].view(batch, 3, 1, 1)
    depth_cam = destination[:, 2:3]
    projected = torch.einsum("bij,bjhw->bihw", intrinsics_dst, destination / depth_cam.clamp(min=1e-6))
    target_u = projected[:, 0]
    target_v = projected[:, 1]
    flow = torch.stack([target_u - pixels[:, 0], target_v - pixels[:, 1]], dim=1)
    x_norm = target_u / max(width - 1, 1) * 2.0 - 1.0
    y_norm = target_v / max(height - 1, 1) * 2.0 - 1.0
    sample_grid = torch.stack([x_norm, y_norm], dim=-1)
    measured = F.grid_sample(depth_dst, sample_grid, align_corners=True, padding_mode="zeros")
    inside = (target_u >= 0) & (target_u <= (width - 1)) & (target_v >= 0) & (target_v <= (height - 1))
    valid = (
        (depth_src > 1e-4)
        & (depth_cam > 1e-4)
        & inside.unsqueeze(1)
        & (measured > 1e-4)
        & ((depth_cam - measured).abs() <= occlusion_m)
    )
    return flow.detach(), valid.detach()
