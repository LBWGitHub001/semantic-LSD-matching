"""
LineFeat Line Segment Matcher.

Performs cross-image line segment matching using descriptor similarity,
mutual nearest neighbor (MNN), Lowe's ratio test, and semantic category gating.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
import torch


def get_distinct_colors(num_colors: int) -> list[tuple[int, int, int]]:
    """Generate visually distinct BGR colors for line matching visualization."""
    colors: list[tuple[int, int, int]] = []
    for i in range(num_colors):
        hue = int(180 * (i / max(num_colors, 1)))
        hsv_pixel = np.uint8([[[hue, 230, 240]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


class LineMatcher:
    """
    Line Segment Matcher with descriptor distance, mutual check, and semantic filtering.
    """

    def __init__(
        self,
        ratio_threshold: float = 0.85,
        mutual_check: bool = True,
        max_angle_diff_deg: float | None = None,
    ) -> None:
        self.ratio_threshold = ratio_threshold
        self.mutual_check = mutual_check
        self.max_angle_diff_deg = max_angle_diff_deg

    def match(
        self,
        desc1: torch.Tensor | np.ndarray,
        desc2: torch.Tensor | np.ndarray,
        lines1: np.ndarray | None = None,
        lines2: np.ndarray | None = None,
        semantics1: Sequence[str | int] | None = None,
        semantics2: Sequence[str | int] | None = None,
    ) -> list[tuple[int, int, float]]:
        """
        Match line segments between two views.

        Args:
            desc1: (N1, D) normalized descriptors of view 1
            desc2: (N2, D) normalized descriptors of view 2
            lines1: Optional (N1, 4) line coordinates [x1, y1, x2, y2]
            lines2: Optional (N2, 4) line coordinates [x1, y1, x2, y2]
            semantics1: Optional (N1,) semantic category names or IDs for lines 1
            semantics2: Optional (N2,) semantic category names or IDs for lines 2

        Returns:
            matches: List of (idx1, idx2, similarity_score)
        """
        if isinstance(desc1, np.ndarray):
            d1 = torch.from_numpy(desc1).float()
        else:
            d1 = desc1.float()

        if isinstance(desc2, np.ndarray):
            d2 = torch.from_numpy(desc2).float()
        else:
            d2 = desc2.float()

        N1, N2 = d1.shape[0], d2.shape[0]
        if N1 == 0 or N2 == 0:
            return []

        # Cosine similarity matrix (N1, N2)
        sim = d1 @ d2.T

        # 1. Semantic Hard Gating (prevent cross-class false matches)
        if semantics1 is not None and semantics2 is not None:
            mask = np.zeros((N1, N2), dtype=bool)
            for i, s1 in enumerate(semantics1):
                for j, s2 in enumerate(semantics2):
                    if s1 != s2 and s1 != "void" and s2 != "void":
                        mask[i, j] = True
            sim[mask] = -1.0

        # 2. Geometric Angle Filtering (optional)
        if self.max_angle_diff_deg is not None and lines1 is not None and lines2 is not None:
            dx1 = lines1[:, 2] - lines1[:, 0]
            dy1 = lines1[:, 3] - lines1[:, 1]
            ang1 = np.arctan2(dy1, dx1) % np.pi

            dx2 = lines2[:, 2] - lines2[:, 0]
            dy2 = lines2[:, 3] - lines2[:, 1]
            ang2 = np.arctan2(dy2, dx2) % np.pi

            max_rad = math.radians(self.max_angle_diff_deg)
            for i in range(N1):
                diff = np.abs(ang1[i] - ang2)
                diff = np.minimum(diff, np.pi - diff)
                invalid = diff > max_rad
                sim[i, invalid] = -1.0

        sim_np = sim.cpu().numpy()

        # 3. Ratio Test & Mutual Nearest Neighbor
        matches: list[tuple[int, int, float]] = []

        # Best in 1 -> 2
        best2 = np.argmax(sim_np, axis=1)  # (N1,)
        best_sim = np.max(sim_np, axis=1)

        # Best in 2 -> 1
        best1 = np.argmax(sim_np, axis=0)  # (N2,)

        for i in range(N1):
            j = best2[i]
            s = best_sim[i]
            if s <= 0.0:
                continue

            # Mutual check
            if self.mutual_check and best1[j] != i:
                continue

            # Ratio test (distance ratio: (1 - s1) / (1 - s2))
            if self.ratio_threshold < 1.0 and N2 > 1:
                row = sim_np[i].copy()
                row[j] = -1.0
                second_s = np.max(row)
                d1_dist = 1.0 - s
                d2_dist = 1.0 - second_s
                if d1_dist > self.ratio_threshold * d2_dist:
                    continue

            matches.append((i, int(j), float(s)))

        matches.sort(key=lambda m: m[2], reverse=True)
        return matches

    @staticmethod
    def draw_matches(
        image1: np.ndarray,
        image2: np.ndarray,
        lines1: np.ndarray,
        lines2: np.ndarray,
        matches: list[tuple[int, int, float]],
        max_draw: int = 40,
        show_connecting_lines: bool = True,
    ) -> np.ndarray:
        """Draw line matching visualization side-by-side."""
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        canvas = np.hstack([image1.copy(), image2.copy()])

        draw_m = matches[:max_draw]
        colors = get_distinct_colors(len(draw_m))

        midpoints1: list[tuple[int, int]] = []
        midpoints2: list[tuple[int, int]] = []

        for k, (i, j, s) in enumerate(draw_m):
            col = colors[k]
            # Line in image 1
            x1, y1, x2, y2 = [int(round(v)) for v in lines1[i]]
            cv2.line(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
            cv2.circle(canvas, (x1, y1), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x2, y2), 4, (255, 255, 255), -1, cv2.LINE_AA)
            mid1 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            midpoints1.append(mid1)

            # Line in image 2 (shifted by w1)
            x3, y3, x4, y4 = [int(round(v)) for v in lines2[j]]
            cv2.line(canvas, (x3 + w1, y3), (x4 + w1, y4), col, 3, cv2.LINE_AA)
            cv2.circle(canvas, (x3 + w1, y3), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x4 + w1, y4), 4, (255, 255, 255), -1, cv2.LINE_AA)
            mid2 = (int((x3 + x4) / 2) + w1, int((y3 + y4) / 2))
            midpoints2.append(mid2)

        if show_connecting_lines:
            overlay = canvas.copy()
            for k in range(len(midpoints1)):
                cv2.line(overlay, midpoints1[k], midpoints2[k], colors[k], 1, cv2.LINE_AA)
            canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)

        # Header bar
        header = np.zeros((50, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"Line Matches: {len(matches)} pairs | Displayed top {len(draw_m)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return np.vstack([header, canvas])
