"""Draw descriptor matches between two letter views."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def match_accuracy(
    descriptor0: torch.Tensor,
    descriptor1: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Nearest-neighbor accuracy when points0[i] is the true match of points1[i]."""
    if points0.shape[0] == 0:
        return 0.0, torch.zeros(0, dtype=torch.long, device=points0.device)
    sampled0 = _sample_descriptors(descriptor0, points0)
    sampled1 = _sample_descriptors(descriptor1, points1)
    choice = torch.matmul(sampled0, sampled1.transpose(0, 1)).argmax(dim=1)
    target = torch.arange(choice.shape[0], device=choice.device)
    accuracy = float((choice == target).float().mean())
    return accuracy, choice


def save_point_match_figure(
    path: Path | str,
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    choice: np.ndarray,
) -> None:
    """Green lines are correct nearest neighbors. Red lines point at the wrong point."""
    left = np.ascontiguousarray(bgr0)
    right = np.ascontiguousarray(bgr1)
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    _, width = left.shape[:2]
    canvas = np.concatenate([left, right], axis=1)
    correct = 0
    for index, (source, target) in enumerate(zip(points0, points1)):
        predicted = int(choice[index]) if index < len(choice) else -1
        hit = predicted == index
        correct += int(hit)
        color = (40, 200, 40) if hit else (40, 40, 220)
        start = (int(round(float(source[0]))), int(round(float(source[1]))))
        end_point = points1[predicted] if 0 <= predicted < len(points1) else target
        end = (int(round(float(end_point[0]))) + width, int(round(float(end_point[1]))))
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (end[0], end[1]), 3, color, -1, cv2.LINE_AA)
    total = len(points0)
    cv2.rectangle(canvas, (0, 0), (168, 28), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"{correct}/{total}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)


def _sample_descriptors(descriptor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """descriptor is (1, C, H, W) or (C, H, W). points is (K, 2) in pixels."""
    if descriptor.dim() == 3:
        descriptor = descriptor.unsqueeze(0)
    _, _, height, width = descriptor.shape
    x = points[:, 0] / max(width - 1, 1) * 2.0 - 1.0
    y = points[:, 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(descriptor, grid, align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
