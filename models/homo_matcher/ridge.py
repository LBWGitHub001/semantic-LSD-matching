"""Turn a DeepLSD distance field into ridge centerlines.

The bright band is a neighborhood. The line used by the loss is the center
of that band: a non-maximum along the line normal, closer than ``thresh``
pixels, then linked with the angle field.
"""

from __future__ import annotations

import numpy as np

RIDGE_THRESH = 1.6
DF_MAX = 5.0


def pack_input(bgr: np.ndarray, df: np.ndarray, angle: np.ndarray, thresh: float = RIDGE_THRESH) -> np.ndarray:
    """Five channels: gray, distance/5, sin(2θ), cos(2θ), ridge mask.

    Raw angle jumps at 0 and π. sin(2θ) and cos(2θ) are continuous on an
    undirected line. The fifth channel marks pixels inside the bright band.
    """
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    distance = np.clip(np.nan_to_num(np.asarray(df, dtype=np.float32), nan=DF_MAX), 0.0, DF_MAX)
    direction = np.mod(np.nan_to_num(np.asarray(angle, dtype=np.float32), nan=0.0), np.pi)
    channels = np.stack(
        [
            gray,
            distance / DF_MAX,
            np.sin(2.0 * direction),
            np.cos(2.0 * direction),
            (distance < thresh).astype(np.float32),
        ],
        axis=0,
    )
    return channels.astype(np.float32)


def ridge_mask(df: np.ndarray, angle: np.ndarray, thresh: float = RIDGE_THRESH) -> np.ndarray:
    """Pixels on the center of a distance-field ridge."""
    distance = np.nan_to_num(np.asarray(df, dtype=np.float64), nan=DF_MAX, posinf=DF_MAX)
    direction = np.mod(np.nan_to_num(np.asarray(angle, dtype=np.float64), nan=0.0), np.pi)
    normal_x = np.rint(-np.sin(direction)).astype(np.int32)
    normal_y = np.rint(np.cos(direction)).astype(np.int32)
    degenerate = (normal_x == 0) & (normal_y == 0)
    normal_x = np.where(degenerate, 1, normal_x)
    padded = np.pad(distance, 1, constant_values=1e6)
    height, width = distance.shape
    ys, xs = np.mgrid[0:height, 0:width]
    left = padded[ys + 1 + normal_y, xs + 1 + normal_x]
    right = padded[ys + 1 - normal_y, xs + 1 - normal_x]
    return (distance < thresh) & (distance <= left) & (distance <= right)


def _endpoints(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, int(np.argmax(values))]
    projection = centered @ direction
    start = center + float(projection.min()) * direction
    end = center + float(projection.max()) * direction
    return start, end, float(np.linalg.norm(end - start))


def extract_ridge_lines(
    df: np.ndarray,
    angle: np.ndarray,
    thresh: float = RIDGE_THRESH,
    min_length: float = 16.0,
    min_pixels: int = 12,
    sample_points: int = 16,
    max_lines: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoints ``(M, 2, 2)`` and uniform samples ``(M, N, 2)`` in pixels.

    Ridges are split into four undirected direction bins so an L-shaped
    corner does not collapse into one diagonal.
    """
    import cv2

    mask = ridge_mask(df, angle, thresh=thresh)
    direction = np.mod(np.nan_to_num(np.asarray(angle, dtype=np.float64), nan=0.0), np.pi)
    bins = np.floor(direction / (np.pi / 4.0)).astype(np.int32) % 4
    found: list[tuple[float, np.ndarray, np.ndarray]] = []
    for bin_id in range(4):
        component_mask = np.ascontiguousarray((mask & (bins == bin_id)).astype(np.uint8))
        count, labels = cv2.connectedComponents(component_mask, connectivity=8)
        for label in range(1, count):
            ys, xs = np.nonzero(labels == label)
            if len(xs) < min_pixels:
                continue
            points = np.stack([xs, ys], axis=1).astype(np.float64)
            start, end, length = _endpoints(points)
            if length < min_length:
                continue
            samples = np.linspace(start, end, sample_points)
            endpoints = np.stack([start, end], axis=0)
            found.append((length, endpoints, samples))
    found.sort(key=lambda item: item[0], reverse=True)
    found = found[:max_lines]
    if not found:
        empty_ends = np.zeros((0, 2, 2), dtype=np.float32)
        empty_samples = np.zeros((0, sample_points, 2), dtype=np.float32)
        return empty_ends, empty_samples
    endpoints = np.stack([item[1] for item in found], axis=0).astype(np.float32)
    samples = np.stack([item[2] for item in found], axis=0).astype(np.float32)
    return endpoints, samples


def resize_fields(
    bgr: np.ndarray,
    df: np.ndarray,
    angle: np.ndarray,
    depth: np.ndarray | None,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Resize RGB, distance, angle and depth. Angle is resized in sin/cos."""
    import cv2

    bgr_out = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    distance = np.nan_to_num(np.asarray(df, dtype=np.float32), nan=DF_MAX)
    direction = np.mod(np.nan_to_num(np.asarray(angle, dtype=np.float32), nan=0.0), np.pi)
    df_out = cv2.resize(distance, (width, height), interpolation=cv2.INTER_AREA)
    sin_out = cv2.resize(np.sin(2.0 * direction), (width, height), interpolation=cv2.INTER_LINEAR)
    cos_out = cv2.resize(np.cos(2.0 * direction), (width, height), interpolation=cv2.INTER_LINEAR)
    angle_out = (0.5 * np.arctan2(sin_out, cos_out)) % np.pi
    depth_out = None
    if depth is not None:
        depth_out = cv2.resize(np.asarray(depth, dtype=np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return bgr_out, df_out.astype(np.float32), angle_out.astype(np.float32), depth_out
