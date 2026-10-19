"""
LineFeat Line Segment Detector.

Wraps OpenCV LSD (Line Segment Detector) with post-filtering,
length thresholding, angle computation, and uniform sampling along line segments.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np


class LineDetector:
    """
    Line Segment Detector (LSD) wrapper.

    Detects 2D line segments with sub-pixel endpoints and provides geometric utilities.
    """

    def __init__(
        self,
        refine_mode: int = cv2.LSD_REFINE_STD,
        scale: float = 0.8,
        sigma_scale: float = 0.6,
        quant: float = 2.0,
        ang_th: float = 22.5,
        log_eps: float = 0.0,
        density_th: float = 0.7,
        n_bins: int = 1024,
    ) -> None:
        self.lsd = cv2.createLineSegmentDetector(
            refine_mode,
            scale,
            sigma_scale,
            quant,
            ang_th,
            log_eps,
            density_th,
            n_bins,
        )

    def detect(
        self,
        image: np.ndarray,
        min_length: float = 20.0,
        margin: int = 5,
    ) -> np.ndarray:
        """
        Detect line segments in image.

        Args:
            image: (H, W, 3) BGR or (H, W) grayscale
            min_length: Minimum pixel length of line segment to keep
            margin: Border margin in pixels to discard border artifacts

        Returns:
            lines: (N, 4) float32 array where each row is [x1, y1, x2, y2]
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape[:2]
        raw_lines, width, prec, nfa = self.lsd.detect(gray)
        if raw_lines is None or len(raw_lines) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        # raw_lines can be (N, 1, 4) or (N, 4) depending on OpenCV version
        lines = raw_lines.reshape(-1, 4)  # (N, 4)

        # Length filtering
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.hypot(dx, dy)
        valid_len = lengths >= min_length

        # Margin filtering (avoid image boundary cuts)
        valid_x = (
            (lines[:, 0] >= margin)
            & (lines[:, 0] < w - margin)
            & (lines[:, 2] >= margin)
            & (lines[:, 2] < w - margin)
        )
        valid_y = (
            (lines[:, 1] >= margin)
            & (lines[:, 1] < h - margin)
            & (lines[:, 3] >= margin)
            & (lines[:, 3] < h - margin)
        )

        valid = valid_len & valid_x & valid_y
        filtered = lines[valid]

        # Sort by length descending
        if len(filtered) > 0:
            sub_lengths = lengths[valid]
            order = np.argsort(-sub_lengths)
            filtered = filtered[order]

        return filtered.astype(np.float32)

    @staticmethod
    def sample_points(lines: np.ndarray, num_samples: int = 5) -> np.ndarray:
        """
        Equidistantly sample points along each line segment.

        Args:
            lines: (N, 4) [x1, y1, x2, y2]
            num_samples: Number of points along each line (e.g. 5: 0%, 25%, 50%, 75%, 100%)

        Returns:
            points: (N, num_samples, 2) [x, y] coordinates
        """
        if len(lines) == 0:
            return np.zeros((0, num_samples, 2), dtype=np.float32)

        ts = np.linspace(0.0, 1.0, num_samples).reshape(1, num_samples, 1)  # (1, K, 1)
        p1 = lines[:, 0:2].reshape(-1, 1, 2)  # (N, 1, 2)
        p2 = lines[:, 2:4].reshape(-1, 1, 2)  # (N, 1, 2)

        # Q(t) = (1 - t) * p1 + t * p2
        pts = (1.0 - ts) * p1 + ts * p2  # (N, K, 2)
        return pts.astype(np.float32)

    @staticmethod
    def compute_angles(lines: np.ndarray) -> np.ndarray:
        """
        Compute angle of line segments in radians [-pi, pi].

        Args:
            lines: (N, 4) [x1, y1, x2, y2]
        Returns:
            angles: (N,) radians
        """
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        return np.arctan2(dy, dx)

    @staticmethod
    def draw_lines(
        image: np.ndarray,
        lines: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detected line segments on image."""
        vis = image.copy()
        for line in lines:
            x1, y1, x2, y2 = [int(round(v)) for v in line]
            cv2.line(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            cv2.circle(vis, (x1, y1), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (x2, y2), 3, (255, 0, 0), -1, cv2.LINE_AA)
        return vis
