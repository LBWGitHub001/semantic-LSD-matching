"""Shared encoder, cross-attention, and a differentiable 8-point homography."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldEncoder(nn.Module):
    """Three stride-2 convolutions. Output resolution is 1/8 of the input."""

    def __init__(self, in_channels: int = 5, feat_dim: int = 64) -> None:
        super().__init__()
        hidden = max(feat_dim // 2, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.stem(image)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.norm_q(query), self.norm_kv(context), self.norm_kv(context), need_weights=False)
        query = query + attended
        return query + self.ff(self.norm_ff(query))


def _grid_tokens(feat: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a regular grid of features and their pixel coordinates."""
    batch, _, _, _ = feat.shape
    device = feat.device
    dtype = feat.dtype
    axis = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).view(1, grid_size * grid_size, 1, 2)
    grid = grid.expand(batch, -1, -1, -1)
    sampled = F.grid_sample(feat, grid, align_corners=True)
    tokens = sampled.squeeze(-1).transpose(1, 2).contiguous()
    return tokens, grid.squeeze(2)


def _pixels_from_norm(norm_xy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    x = (norm_xy[..., 0] + 1.0) * 0.5 * (width - 1)
    y = (norm_xy[..., 1] + 1.0) * 0.5 * (height - 1)
    return torch.stack([x, y], dim=-1)


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
        return torch.eye(3, device=src.device, dtype=dtype).expand(batch, -1, -1).clone()
    homography_n = src.new_zeros(batch, 3, 3)
    homography_n[:, 0, 0] = solved[:, 0]
    homography_n[:, 0, 1] = solved[:, 1]
    homography_n[:, 0, 2] = solved[:, 2]
    homography_n[:, 1, 0] = solved[:, 3]
    homography_n[:, 1, 1] = solved[:, 4]
    homography_n[:, 1, 2] = solved[:, 5]
    homography_n[:, 2, 0] = solved[:, 6]
    homography_n[:, 2, 1] = solved[:, 7]
    homography_n[:, 2, 2] = 1.0
    scale = torch.zeros(batch, 3, 3, device=src.device, dtype=dtype)
    scale[:, 0, 0] = span
    scale[:, 1, 1] = span
    scale[:, 2, 2] = 1.0
    scale_inv = torch.zeros_like(scale)
    scale_inv[:, 0, 0] = 1.0 / span
    scale_inv[:, 1, 1] = 1.0 / span
    scale_inv[:, 2, 2] = 1.0
    homography = scale @ homography_n @ scale_inv
    homography = homography / homography[:, 2, 2].clamp(min=1e-6)[:, None, None]
    return homography


class HomoMatcher(nn.Module):
    """Two 5-channel views in, two homographies out.

    A shared encoder reads gray, distance, sin(2θ), cos(2θ) and the ridge
    mask. Cross-attention mixes the views. Each grid point predicts where it
    lands in the other view, and a weighted DLT turns those matches into
    H12 and H21. h33 is fixed to 1.
    """

    def __init__(
        self,
        feat_dim: int = 64,
        d_model: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        grid_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.grid_size = int(grid_size)
        self.encoder = FieldEncoder(5, feat_dim)
        self.proj = nn.Linear(feat_dim, d_model)
        self.cross_attn = nn.ModuleList(CrossAttentionBlock(d_model, nhead) for _ in range(num_layers))
        self.coord_head = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )
        self.conf_head = nn.Linear(d_model + 2, 1)
        nn.init.zeros_(self.coord_head[-1].weight)
        nn.init.zeros_(self.coord_head[-1].bias)

    def _matches(self, tokens: torch.Tensor, norm_xy: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = _pixels_from_norm(norm_xy, width, height)
        head_in = torch.cat([tokens, norm_xy], dim=-1)
        delta = torch.tanh(self.coord_head(head_in))
        offset = delta * head_in.new_tensor([width, height])
        destination = (source + offset).clamp(min=0)
        destination = torch.stack(
            [
                destination[..., 0].clamp(max=width - 1),
                destination[..., 1].clamp(max=height - 1),
            ],
            dim=-1,
        )
        confidence = torch.sigmoid(self.conf_head(head_in).squeeze(-1))
        return source, destination, confidence

    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> dict[str, torch.Tensor]:
        feat0 = self.encoder(image0)
        feat1 = self.encoder(image1)
        tokens0, norm0 = _grid_tokens(feat0, self.grid_size)
        tokens1, norm1 = _grid_tokens(feat1, self.grid_size)
        tokens0 = self.proj(tokens0)
        tokens1 = self.proj(tokens1)
        for block in self.cross_attn:
            updated0 = block(tokens0, tokens1)
            updated1 = block(tokens1, tokens0)
            tokens0, tokens1 = updated0, updated1
        height, width = image0.shape[-2:]
        src0, dst0, conf0 = self._matches(tokens0, norm0, height, width)
        src1, dst1, conf1 = self._matches(tokens1, norm1, height, width)
        ridge0 = F.grid_sample(image0[:, 4:5], norm0.unsqueeze(2), align_corners=True).squeeze(1).squeeze(-1)
        ridge1 = F.grid_sample(image1[:, 4:5], norm1.unsqueeze(2), align_corners=True).squeeze(1).squeeze(-1)
        weight0 = conf0 * (0.25 + ridge0)
        weight1 = conf1 * (0.25 + ridge1)
        return {
            "H12": weighted_dlt(src0, dst0, weight0),
            "H21": weighted_dlt(src1, dst1, weight1),
            "conf0": conf0,
            "conf1": conf1,
        }
