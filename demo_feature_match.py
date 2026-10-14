"""
Demo: classical feature matching (ORB).

Usage (from project root):
  python demo_feature_match.py
  python demo_feature_match.py path/to/img1.jpg path/to/img2.jpg
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from feature_match.matcher import draw_matches, match_images, warp_for_demo

SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORB feature matching demo")
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
        default=Path("outputs/feature_match"),
        help="Output directory (default: outputs/feature_match)",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=80,
        help="Max number of matches to draw (default: 80)",
    )
    return parser.parse_args()


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def prepare_pair(
    args: argparse.Namespace, out_dir: Path
) -> tuple[np.ndarray, np.ndarray, str]:
    if len(args.images) >= 2:
        return load_bgr(args.images[0]), load_bgr(args.images[1]), "custom"
    if len(args.images) == 1:
        img1 = load_bgr(args.images[0])
        img2 = warp_for_demo(img1)
        return img1, img2, "warped_from_single"

    sample_path = out_dir / "sample.jpg"
    if not sample_path.is_file():
        print(f"downloading sample -> {sample_path}")
        urllib.request.urlretrieve(SAMPLE_URL, sample_path)
    img1 = load_bgr(sample_path)
    img2 = warp_for_demo(img1)
    cv2.imwrite(str(out_dir / "sample_warped.jpg"), img2)
    return img1, img2, "sample_warped"


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    img1, img2, mode = prepare_pair(args, out_dir)
    result = match_images(img1, img2)
    vis = draw_matches(img1, img2, result, max_draw=args.max_draw)

    out_path = out_dir / "matches.jpg"
    cv2.imwrite(str(out_path), vis)

    print(f"mode={mode}")
    print(f"keypoints: img1={len(result.keypoints1)}, img2={len(result.keypoints2)}")
    print(f"good matches: {len(result.matches)}")
    print(f"drew top: {min(args.max_draw, len(result.matches))}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
