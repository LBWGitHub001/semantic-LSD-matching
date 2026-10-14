"""ORB keypoint detection + Brute-Force matching."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    keypoints1: list
    keypoints2: list
    matches: list
    image1_gray: np.ndarray
    image2_gray: np.ndarray


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def match_images(
    image1_bgr: np.ndarray,
    image2_bgr: np.ndarray,
    max_features: int = 2000,
    ratio: float = 0.75,
) -> MatchResult:
    """
    Detect ORB features on both images and match with Lowe ratio test.

    Returns keypoints and good matches for visualization.
    """
    gray1 = _to_gray(image1_bgr)
    gray2 = _to_gray(image2_bgr)

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        return MatchResult(keypoints1 or [], keypoints2 or [], [], gray1, gray2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good: list = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    good.sort(key=lambda m: m.distance)
    return MatchResult(keypoints1, keypoints2, good, gray1, gray2)


def draw_matches(
    image1_bgr: np.ndarray,
    image2_bgr: np.ndarray,
    result: MatchResult,
    max_draw: int = 80,
) -> np.ndarray:
    """Draw top matches side-by-side (OpenCV drawMatches)."""
    matches = result.matches[:max_draw]
    return cv2.drawMatches(
        image1_bgr,
        result.keypoints1,
        image2_bgr,
        result.keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def warp_for_demo(image_bgr: np.ndarray) -> np.ndarray:
    """Create a second view by rotating/scaling/translating the first image."""
    h, w = image_bgr.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle=25.0, scale=0.85)
    matrix[0, 2] += w * 0.05
    matrix[1, 2] += h * 0.02
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
