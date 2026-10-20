"""
Geometric Field Generator for Line Segment Matching.

Computes continuous spatial geometry representations (EDT + AFM) from 2D line segments:
  - Channel 0: Normalized Euclidean Distance Transform (EDT) in [0, 1]
  - Channel 1: Attraction / Orientation Field sin(2*theta)
  - Channel 2: Attraction / Orientation Field cos(2*theta)

When combined with RGB images (3 channels), forms a 6-channel dense spatial input tensor.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def compute_geometry_field_np(
    lines: np.ndarray | list,
    height: int = 720,
    width: int = 1280,
    max_dist: float = 60.0,
    line_thickness: int = 3,
) -> np.ndarray:
    """
    Computes a 3-channel continuous geometry field from 2D line segments using OpenCV.

    Args:
        lines: (N, 4) array of [x1, y1, x2, y2] line segments.
        height: Image height.
        width: Image width.
        max_dist: Distance clipping threshold for EDT normalization.
        line_thickness: Pixel thickness to rasterize line segments.

    Returns:
        (3, height, width) float32 numpy array: [EDT_norm, sin2theta, cos2theta].
    """
    lines = np.asarray(lines, dtype=np.float32)
    line_mask = np.zeros((height, width), dtype=np.uint8)
    sin_field = np.zeros((height, width), dtype=np.float32)
    cos_field = np.zeros((height, width), dtype=np.float32)

    if len(lines) == 0:
        # No lines: uniform max distance (1.0) and zero angles
        return np.stack([
            np.ones((height, width), dtype=np.float32),
            sin_field,
            cos_field,
        ], axis=0)

    # 1. Rasterize lines with continuous angle signatures
    for line in lines:
        x1, y1, x2, y2 = line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = np.hypot(dx, dy)
        if length < 1e-3:
            continue

        theta = np.arctan2(dy, dx)
        s2 = float(np.sin(2.0 * theta))
        c2 = float(np.cos(2.0 * theta))

        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))

        cv2.line(line_mask, pt1, pt2, 255, line_thickness)
        cv2.line(sin_field, pt1, pt2, s2, line_thickness)
        cv2.line(cos_field, pt1, pt2, c2, line_thickness)

    # 2. Euclidean Distance Transform via OpenCV C++
    inv_mask = 255 - line_mask
    edt = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    edt_norm = np.clip(edt / max_dist, 0.0, 1.0).astype(np.float32)

    # 3. Diffuse angle signatures across nearby distance
    # Damping weight: 1.0 on line, exponentially decaying with distance
    decay_scale = max_dist * 0.25
    weight = np.exp(-edt / decay_scale).astype(np.float32)

    sin_diffused = sin_field * weight
    cos_diffused = cos_field * weight

    return np.stack([edt_norm, sin_diffused, cos_diffused], axis=0)


def create_6ch_input_tensor(
    rgb_tensor: torch.Tensor,
    lines: np.ndarray | torch.Tensor | list,
    max_dist: float = 60.0,
) -> torch.Tensor:
    """
    Concatenates a (B, 3, H, W) or (3, H, W) normalized RGB tensor with its (3, H, W)
    continuous geometric field (EDT + AFM), producing a (B, 6, H, W) input tensor.

    Args:
        rgb_tensor: (3, H, W) or (B, 3, H, W) normalized RGB float tensor.
        lines: (N, 4) or list of (N_b, 4) for batch.
        max_dist: Max distance parameter for EDT.

    Returns:
        (B, 6, H, W) float32 tensor on the same device as rgb_tensor.
    """
    is_batched = (rgb_tensor.ndim == 4)
    if not is_batched:
        rgb_tensor = rgb_tensor.unsqueeze(0)

    B, C, H, W = rgb_tensor.shape
    device = rgb_tensor.device

    geom_fields = []
    if isinstance(lines, torch.Tensor):
        lines_np = lines.detach().cpu().numpy()
    else:
        lines_np = lines

    for b in range(B):
        if is_batched and isinstance(lines_np, (list, tuple)):
            b_lines = lines_np[b]
        elif is_batched and isinstance(lines_np, np.ndarray) and lines_np.ndim == 3:
            b_lines = lines_np[b]
        else:
            b_lines = lines_np

        field = compute_geometry_field_np(b_lines, height=H, width=W, max_dist=max_dist)
        geom_fields.append(field)

    geom_tensor = torch.from_numpy(np.stack(geom_fields, axis=0)).to(device=device, dtype=rgb_tensor.dtype)
    six_ch = torch.cat([rgb_tensor, geom_tensor], dim=1)

    return six_ch if is_batched else six_ch.squeeze(0)
