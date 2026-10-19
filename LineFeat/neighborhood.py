"""
LineFeat Line Neighborhood Alignment & Warped Patch Matching.

Implements line neighborhood transformation, canonical ribbon extraction,
direct patch warping between line pairs, and similarity verification.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_line_geometry(line: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Compute geometric properties of a 2D line segment [x1, y1, x2, y2].

    Returns:
        midpoint: (2,) [cx, cy]
        length: float
        angle: float in radians [-pi, pi]
        tangent: (2,) unit vector along line
        normal: (2,) unit vector perpendicular to line
    """
    x1, y1, x2, y2 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    midpoint = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    dx = x2 - x1
    dy = y2 - y1
    length = max(math.hypot(dx, dy), 1e-4)
    tangent = np.array([dx / length, dy / length], dtype=np.float32)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    angle = math.atan2(dy, dx)
    return midpoint, length, angle, tangent, normal


class LineNeighborhoodTransformer:
    """
    Extracts canonical line neighborhood strips and computes affine warp between line segments.
    """

    def __init__(
        self,
        strip_width: int = 128,  # Pixel resolution along line length
        strip_height: int = 32,  # Pixel resolution across normal direction
        band_ratio: float = 0.25, # Band width as ratio of line length or fixed pixels
        fixed_band_pixels: float | None = 24.0, # Perpendicular support region thickness in pixels
    ) -> None:
        self.strip_w = strip_width
        self.strip_h = strip_height
        self.band_ratio = band_ratio
        self.fixed_band_pixels = fixed_band_pixels

    def get_canonical_affine_matrix(
        self,
        line: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """
        Compute affine transformation matrix between image coordinates and canonical strip coordinates.
        Strip coordinates: x in [0, strip_w-1], y in [0, strip_h-1].
        """
        mid, length, _, tangent, normal = get_line_geometry(line)
        band_h = self.fixed_band_pixels if self.fixed_band_pixels is not None else length * self.band_ratio
        band_h = max(band_h, 8.0)

        # 3 correspondence points in image:
        # Point 0: Line start p1
        p_start = mid - 0.5 * length * tangent
        # Point 1: Line end p2
        p_end = mid + 0.5 * length * tangent
        # Point 2: Shifted along normal direction at midpoint
        p_side = mid + 0.5 * band_h * normal

        # 3 correspondence points in canonical strip:
        # p_start -> (0, strip_h / 2)
        # p_end   -> (strip_w - 1, strip_h / 2)
        # p_side  -> (strip_w / 2, strip_h - 1)
        src_pts = np.array([p_start, p_end, p_side], dtype=np.float32)
        dst_pts = np.array(
            [
                [0.0, self.strip_h / 2.0],
                [float(self.strip_w - 1), self.strip_h / 2.0],
                [float(self.strip_w / 2.0), float(self.strip_h - 1)],
            ],
            dtype=np.float32,
        )

        if inverse:
            return cv2.getAffineTransform(dst_pts, src_pts)
        return cv2.getAffineTransform(src_pts, dst_pts)

    def extract_canonical_strip(
        self,
        image: np.ndarray,
        line: np.ndarray,
    ) -> np.ndarray:
        """
        Extract canonical rectangular neighborhood strip around a 2D line segment.

        Args:
            image: (H, W, 3) or (H, W) uint8 / float32
            line: [x1, y1, x2, y2]

        Returns:
            strip: (strip_h, strip_w, C) or (strip_h, strip_w) rectified patch
        """
        M = self.get_canonical_affine_matrix(line, inverse=False)
        strip = cv2.warpAffine(
            image,
            M,
            (self.strip_w, self.strip_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return strip

    def warp_line1_to_line2(
        self,
        image1: np.ndarray,
        line1: np.ndarray,
        line2: np.ndarray,
        output_shape: Tuple[int, int] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Directly warp the neighborhood of Line 1 in Image 1 to align with Line 2 in Image 2.

        Returns:
            warped_image1: Image 1 warped into Image 2 coordinate frame
            M_1to2: 2x3 affine transformation matrix from Image 1 to Image 2
        """
        # M_1_to_canon: Image 1 -> Canonical
        M1 = self.get_canonical_affine_matrix(line1, inverse=False)
        # M_canon_to_2: Canonical -> Image 2
        M2_inv = self.get_canonical_affine_matrix(line2, inverse=True)

        # Convert 2x3 to 3x3
        A1 = np.vstack([M1, [0, 0, 1]])
        A2_inv = np.vstack([M2_inv, [0, 0, 1]])

        # Full transform: T_1to2 = A2_inv @ A1
        T_1to2 = A2_inv @ A1
        M_1to2 = T_1to2[:2, :]

        if output_shape is None:
            out_w, out_h = image1.shape[1], image1.shape[0]
        else:
            out_h, out_w = output_shape

        warped = cv2.warpAffine(
            image1,
            M_1to2,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped, M_1to2


def compute_strip_zncc(
    strip1: np.ndarray,
    strip2: np.ndarray,
    check_both_orientations: bool = True,
) -> float:
    """
    Compute Zero-mean Normalized Cross-Correlation (ZNCC) between two neighborhood strips.
    ZNCC ranges in [-1.0, 1.0], where 1.0 indicates perfect linear correlation.
    """
    if strip1.ndim == 3:
        g1 = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        g1 = strip1.astype(np.float32)

    if strip2.ndim == 3:
        g2 = cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        g2 = strip2.astype(np.float32)

    def _zncc(a: np.ndarray, b: np.ndarray) -> float:
        a_z = a - np.mean(a)
        b_z = b - np.mean(b)
        denom = math.sqrt(float(np.sum(a_z * a_z) * np.sum(b_z * b_z)))
        if denom < 1e-6:
            return 0.0
        return float(np.sum(a_z * b_z) / denom)

    score = _zncc(g1, g2)

    if check_both_orientations:
        # Flip 180 degrees (flip both horizontally and vertically)
        g2_flipped = np.flip(g2, axis=(0, 1))
        score_flip = _zncc(g1, g2_flipped)
        return max(score, score_flip)

    return score


def extract_deep_strip_descriptor(
    feature_map: torch.Tensor,
    line: np.ndarray,
    orig_size: Tuple[int, int] = (720, 1280),
    num_samples_length: int = 16,
    num_samples_width: int = 5,
    band_pixels: float = 20.0,
) -> torch.Tensor:
    """
    Extract a 2D grid descriptor across the line strip from deep feature map.

    Args:
        feature_map: (1, C, H_f, W_f) or (C, H_f, W_f)
        line: [x1, y1, x2, y2]
        orig_size: (H_orig, W_orig)
        num_samples_length: Number of points along the line direction
        num_samples_width: Number of points across normal direction
        band_pixels: Total strip width in original image pixels

    Returns:
        descriptor: (C,) or (C * W,) normalized deep descriptor
    """
    if feature_map.ndim == 3:
        feat = feature_map.unsqueeze(0)
    else:
        feat = feature_map

    device = feat.device
    mid, length, _, tangent, normal = get_line_geometry(line)

    H_orig, W_orig = orig_size
    K_len = num_samples_length
    K_w = num_samples_width

    # Coordinates in line coordinate system
    u_vals = np.linspace(-0.5 * length, 0.5 * length, K_len)  # Along line
    v_vals = np.linspace(-0.5 * band_pixels, 0.5 * band_pixels, K_w)  # Across line

    grid_pts = []
    for v in v_vals:
        row = mid + np.outer(u_vals, tangent) + v * normal  # (K_len, 2)
        grid_pts.append(row)
    grid_pts = np.stack(grid_pts, axis=0)  # (K_w, K_len, 2) in [x, y]

    # Convert to normalized [-1, 1] for grid_sample
    gx = (grid_pts[:, :, 0] / (W_orig - 1.0)) * 2.0 - 1.0
    gy = (grid_pts[:, :, 1] / (H_orig - 1.0)) * 2.0 - 1.0
    grid_t = torch.tensor(np.stack([gx, gy], axis=-1), dtype=torch.float32, device=device).unsqueeze(0)

    # sampled: (1, C, K_w, K_len)
    sampled = F.grid_sample(feat, grid_t, mode="bilinear", padding_mode="border", align_corners=True)[0]
    # Pool across the strip (mean pooling)
    desc = torch.mean(sampled, dim=(1, 2))  # (C,)
    desc = F.normalize(desc, p=2, dim=0)
    return desc
