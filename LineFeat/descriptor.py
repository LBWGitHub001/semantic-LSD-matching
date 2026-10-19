"""
LineFeat Line Segment Descriptor Extractor.

Extracts deep learning line descriptors by uniformly sampling points along
2D line segments from dense convolutional feature maps (e.g., XFeat, ResNet).
Supports mean-pooling, max-pooling, and concatenation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


class LineDescriptor:
    """
    Line Descriptor Extractor.

    Samples points along 2D line segments and pools from deep feature maps.
    """

    def __init__(
        self,
        num_samples: int = 5,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> None:
        """
        Args:
            num_samples: Number of points sampled along each line segment.
            pooling: 'mean', 'max', or 'concat'
            normalize: L2 normalize the resulting line descriptors
        """
        self.num_samples = num_samples
        self.pooling = pooling.lower()
        self.normalize = normalize

    def extract(
        self,
        lines: np.ndarray | torch.Tensor,
        feature_map: torch.Tensor,
        orig_size: tuple[int, int] = (720, 1280),
    ) -> torch.Tensor:
        """
        Extract descriptors for line segments from a dense feature tensor.

        Args:
            lines: (N, 4) [x1, y1, x2, y2] in original image pixel coordinates
            feature_map: (1, C, H_f, W_f) or (C, H_f, W_f) feature tensor
            orig_size: (H_orig, W_orig) size of original image corresponding to line coords

        Returns:
            descriptors: (N, D) float32 L2-normalized line descriptors
        """
        if isinstance(lines, np.ndarray):
            lines_t = torch.from_numpy(lines).float()
        else:
            lines_t = lines.float()

        if len(lines_t) == 0:
            c = feature_map.shape[1] if feature_map.ndim == 4 else feature_map.shape[0]
            d = c * self.num_samples if self.pooling == "concat" else c
            return torch.zeros((0, d), device=feature_map.device, dtype=torch.float32)

        if feature_map.ndim == 3:
            feat = feature_map.unsqueeze(0)  # (1, C, H_f, W_f)
        else:
            feat = feature_map

        device = feat.device
        lines_t = lines_t.to(device)

        N = lines_t.shape[0]
        K = self.num_samples
        H_orig, W_orig = orig_size

        # 1. Sample K points along each line
        ts = torch.linspace(0.0, 1.0, K, device=device).view(1, K, 1)  # (1, K, 1)
        p1 = lines_t[:, 0:2].view(N, 1, 2)  # (N, 1, 2)
        p2 = lines_t[:, 2:4].view(N, 1, 2)  # (N, 1, 2)
        pts = (1.0 - ts) * p1 + ts * p2  # (N, K, 2) [x, y]

        # 2. Normalize to [-1, 1] for torch grid_sample
        grid_x = (pts[:, :, 0] / (W_orig - 1.0)) * 2.0 - 1.0
        grid_y = (pts[:, :, 1] / (H_orig - 1.0)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, K, 2)
        grid = grid.view(1, N, K, 2)  # (1, N, K, 2)

        # 3. Bilinear interpolation from feature map
        # sampled shape: (1, C, N, K)
        sampled = F.grid_sample(
            feat, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0]  # (C, N, K)

        # Transpose to (N, K, C)
        sampled = sampled.permute(1, 2, 0)  # (N, K, C)

        # 4. Pooling
        if self.pooling == "mean":
            desc = torch.mean(sampled, dim=1)  # (N, C)
        elif self.pooling == "max":
            desc, _ = torch.max(sampled, dim=1)  # (N, C)
        elif self.pooling == "concat":
            desc = sampled.reshape(N, -1)  # (N, K * C)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # 5. L2 Normalization
        if self.normalize:
            desc = F.normalize(desc, p=2, dim=-1)

        return desc
