"""Draw ridge endpoints on the original image."""

from __future__ import annotations

import numpy as np


def draw_ridge_lines(bgr: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
    import cv2

    canvas = bgr.copy()
    for segment in endpoints:
        start = (int(round(float(segment[0, 0]))), int(round(float(segment[0, 1]))))
        end = (int(round(float(segment[1, 0]))), int(round(float(segment[1, 1]))))
        cv2.line(canvas, start, end, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(canvas, start, end, (0, 180, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 4, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, end, 4, (255, 255, 0), -1, cv2.LINE_AA)
    return canvas
