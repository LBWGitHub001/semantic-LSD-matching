"""Fully convolutional velocity field."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sanitize_grad(value: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    """Keep the forward value and replace non-finite or huge backward gradients."""

    class _Sanitize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        @staticmethod
        def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            return grad.clamp(-limit, limit)

    return _Sanitize.apply(value)


def weighted_dlt(src: torch.Tensor, dst: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Solve a weighted homography with h33 fixed to 1.

    ``src`` and ``dst`` are pixel coordinates ``(B, N, 2)``. Coordinates are
    normalized to [-1, 1] before the solve and mapped back afterwards.
    """
    batch, _, _ = src.shape
    dtype = torch.float32
    src = src.to(dtype)
    dst = dst.to(dtype)
    weight = weight.to(dtype).clamp(min=0.0)
    span = torch.maximum(src.detach().abs().amax(dim=(1, 2)), dst.detach().abs().amax(dim=(1, 2))).clamp(min=1.0)
    src_n = src / span[:, None, None]
    dst_n = dst / span[:, None, None]
    x, y = src_n[..., 0], src_n[..., 1]
    xp, yp = dst_n[..., 0], dst_n[..., 1]
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    row_x = torch.stack([x, y, ones, zeros, zeros, zeros, -xp * x, -xp * y], dim=-1)
    row_y = torch.stack([zeros, zeros, zeros, x, y, ones, -yp * x, -yp * y], dim=-1)
    design = torch.stack([row_x, row_y], dim=2).reshape(batch, -1, 8)
    target = torch.stack([xp, yp], dim=-1).reshape(batch, -1, 1)
    weights = weight.repeat_interleave(2, dim=1).unsqueeze(-1)
    weighted = design * weights
    normal = weighted.transpose(1, 2) @ design
    rhs = weighted.transpose(1, 2) @ target
    eye = torch.eye(8, device=src.device, dtype=dtype).expand(batch, -1, -1)
    try:
        solved = torch.linalg.solve(normal + eps * eye, rhs).squeeze(-1)
    except RuntimeError:
        # Keep a zero-valued edge into the matches so backward still runs.
        return torch.eye(3, device=src.device, dtype=dtype).expand(batch, -1, -1) + (dst.sum() + weight.sum()) * 0
    ones = torch.ones(batch, 1, device=src.device, dtype=dtype)
    homography_n = torch.stack(
        [
            solved[:, 0:3],
            solved[:, 3:6],
            torch.cat([solved[:, 6:8], ones], dim=-1),
        ],
        dim=1,
    )
    scale_diag = torch.stack([span, span, torch.ones_like(span)], dim=-1)
    scale = torch.diag_embed(scale_diag)
    scale_inv = torch.diag_embed(1.0 / scale_diag)
    homography = scale @ homography_n @ scale_inv
    homography = homography / homography[:, 2, 2].clamp(min=1e-6)[:, None, None]
    return _sanitize_grad(homography)


class DenseFCN(nn.Module):
    """Fully convolutional encoder-decoder. Output resolution matches the input."""

    def __init__(self, in_channels: int = 5, feat_dim: int = 32) -> None:
        super().__init__()
        width = feat_dim

        def block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.down1 = block(in_channels, width, stride=2)
        self.down2 = block(width, width, stride=2)
        self.down3 = block(width, width, stride=2)
        self.fuse2 = block(width * 2, width)
        self.fuse1 = block(width * 2, width)
        self.fuse0 = block(width + in_channels, width)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(image)
        skip2 = self.down2(skip1)
        bottleneck = self.down3(skip2)
        up2 = F.interpolate(bottleneck, size=skip2.shape[-2:], mode="bilinear", align_corners=True)
        up2 = self.fuse2(torch.cat([up2, skip2], dim=1))
        up1 = F.interpolate(up2, size=skip1.shape[-2:], mode="bilinear", align_corners=True)
        up1 = self.fuse1(torch.cat([up1, skip1], dim=1))
        up0 = F.interpolate(up1, size=image.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse0(torch.cat([up0, image], dim=1))


class FCNField(nn.Module):
    """Dense velocity field between two views.

    Direction is atan2(vy, vx) and magnitude is the vector length.
    """

    def __init__(self, feat_dim: int = 32, grid_size: int = 16, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.feat_dim = feat_dim
        self.grid_size = int(grid_size)
        self.fcn = DenseFCN(5, feat_dim)
        self.velocity_head = nn.Conv2d(feat_dim * 2, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def _velocity(self, feat_from: torch.Tensor, feat_to: torch.Tensor) -> torch.Tensor:
        return self.velocity_head(torch.cat([feat_from, feat_to], dim=1))

    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> dict[str, torch.Tensor]:
        feat0 = self.fcn(image0)
        feat1 = self.fcn(image1)
        velocity12 = self._velocity(feat0, feat1)
        velocity21 = self._velocity(feat1, feat0)
        magnitude12 = torch.linalg.vector_norm(velocity12, dim=1)
        magnitude21 = torch.linalg.vector_norm(velocity21, dim=1)
        return {
            "velocity12": velocity12,
            "velocity21": velocity21,
            "magnitude12": magnitude12,
            "magnitude21": magnitude21,
            "direction12": torch.atan2(velocity12[:, 1], velocity12[:, 0]),
            "direction21": torch.atan2(velocity21[:, 1], velocity21[:, 0]),
        }
