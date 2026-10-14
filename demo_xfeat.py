"""
Demo: XFeat (CVPR 2024) feature matching.

Requires official code at:
  third_party/accelerated_features
  (git clone https://github.com/verlab/accelerated_features.git)

Usage (from project root):
  python demo_xfeat.py
  python demo_xfeat.py path/to/img1.jpg path/to/img2.jpg
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
XFEAT_ROOT = ROOT / "third_party" / "accelerated_features"

SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"


def _ensure_xfeat_path() -> None:
    if not XFEAT_ROOT.is_dir():
        raise FileNotFoundError(
            f"XFeat not found at {XFEAT_ROOT}\n"
            "Clone with:\n"
            "  git clone https://github.com/verlab/accelerated_features.git "
            "third_party/accelerated_features"
        )
    weights = XFEAT_ROOT / "weights" / "xfeat.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing weights: {weights}")
    # Official imports are `from modules.xfeat import XFeat`
    sys.path.insert(0, str(XFEAT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XFeat matching demo")
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Optional: image1 [image2]. If omitted, download a sample and warp it.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs/xfeat"),
        help="Output directory (default: outputs/xfeat)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4096,
        help="Max keypoints (default: 4096)",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=120,
        help="Max matches to draw (default: 120)",
    )
    parser.add_argument(
        "--star",
        action="store_true",
        help="Use semi-dense match_xfeat_star instead of sparse match_xfeat",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8 (H,W,3) for XFeat."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def warp_for_demo(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle=25.0, scale=0.85)
    matrix[0, 2] += w * 0.05
    matrix[1, 2] += h * 0.02
    return cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def prepare_pair(
    args: argparse.Namespace, out_dir: Path
) -> tuple[np.ndarray, np.ndarray, str]:
    if len(args.images) >= 2:
        return load_rgb(args.images[0]), load_rgb(args.images[1]), "custom"
    if len(args.images) == 1:
        img1 = load_rgb(args.images[0])
        return img1, warp_for_demo(img1), "warped_from_single"

    sample_path = out_dir / "sample.jpg"
    if not sample_path.is_file():
        print(f"downloading sample -> {sample_path}")
        urllib.request.urlretrieve(SAMPLE_URL, sample_path)
    img1 = load_rgb(sample_path)
    img2 = warp_for_demo(img1)
    cv2.imwrite(
        str(out_dir / "sample_warped.jpg"),
        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
    )
    return img1, img2, "sample_warped"


def draw_mkpts(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    max_draw: int,
) -> np.ndarray:
    n = min(len(mkpts0), max_draw)
    mkpts0 = mkpts0[:n]
    mkpts1 = mkpts1[:n]

    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in mkpts0]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in mkpts1]
    matches = [cv2.DMatch(i, i, 0) for i in range(n)]

    bgr1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
    bgr2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
    return cv2.drawMatches(
        bgr1,
        kp1,
        bgr2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def main() -> None:
    args = parse_args()
    if args.cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    _ensure_xfeat_path()
    from modules.xfeat import XFeat  # noqa: E402

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    img1, img2, mode = prepare_pair(args, out_dir)
    print(f"mode={mode}, loading XFeat (top_k={args.top_k}) ...")
    xfeat = XFeat(top_k=args.top_k)

    if args.star:
        mkpts0, mkpts1 = xfeat.match_xfeat_star(img1, img2, top_k=args.top_k)
    else:
        mkpts0, mkpts1 = xfeat.match_xfeat(img1, img2, top_k=args.top_k)

    vis = draw_mkpts(img1, img2, mkpts0, mkpts1, max_draw=args.max_draw)
    out_path = out_dir / "matches.jpg"
    cv2.imwrite(str(out_path), vis)

    print(f"matches: {len(mkpts0)}")
    print(f"drew top: {min(args.max_draw, len(mkpts0))}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
