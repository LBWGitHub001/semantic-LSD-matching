#!/usr/bin/env python3
"""Extract line endpoints from a DeepLSD distance field and draw them.

The endpoints come only from the ridge centers of the distance field and the
direction in the angle field. CAD lines and LSD are not used.

  python scripts/test_deeplsd_lines.py path/to/rgb_rawlight.png
  python scripts/test_deeplsd_lines.py view0.png --peer view1.png --depth0 depth0.png --depth1 depth1.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.homo_matcher.draw import draw_ridge_lines
from models.homo_matcher.fields import DEFAULT_WEIGHT, infer_fields, load_deeplsd
from models.homo_matcher.ridge import extract_ridge_lines


def _load_fields(rgb_path: Path, fields_path: Path | None, net, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    if fields_path is not None:
        stored = np.load(fields_path)
        return bgr, stored["df"].astype(np.float32), stored["angle"].astype(np.float32)
    distance, angle = infer_fields(net, bgr, device)
    return bgr, distance, angle


def _profile_figure(
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    samples0: np.ndarray,
    depth0: np.ndarray,
    depth1: np.ndarray,
    dest: Path,
) -> None:
    """Draw one ridge and the two depth sequences it would compare under H = I."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).is_file():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    if len(samples0) == 0:
        return
    lengths = np.linalg.norm(samples0[:, -1] - samples0[:, 0], axis=1)
    choice = samples0[int(np.argmax(lengths))]
    xs = np.clip(np.rint(choice[:, 0]).astype(int), 0, depth0.shape[1] - 1)
    ys = np.clip(np.rint(choice[:, 1]).astype(int), 0, depth0.shape[0] - 1)
    series0 = depth0[ys, xs].astype(np.float64)
    ys1 = np.clip(ys, 0, depth1.shape[0] - 1)
    xs1 = np.clip(xs, 0, depth1.shape[1] - 1)
    series1 = depth1[ys1, xs1].astype(np.float64)

    def _zscore(values: np.ndarray) -> np.ndarray:
        valid = values > 1e-4
        if valid.sum() < 2:
            return np.zeros_like(values)
        mean = values[valid].mean()
        std = max(float(values[valid].std()), 1e-3)
        return (values - mean) / std

    canvas0 = bgr0.copy()
    points = np.round(choice).astype(int)
    cv2.polylines(canvas0, [points], False, (0, 180, 255), 2, cv2.LINE_AA)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cv2.cvtColor(canvas0, cv2.COLOR_BGR2RGB))
    axes[0].set_title("亮脊采样点")
    axes[0].axis("off")
    axis = np.arange(len(choice))
    axes[1].plot(axis, _zscore(series0), label="图1深度")
    axes[1].plot(axis, _zscore(series1), label="图2同一位置")
    axes[1].set_title("归一化后的两条深度序列")
    axes[1].set_xlabel("沿脊的采样序号")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(dest, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw DeepLSD ridge endpoints on the photo.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--fields", type=Path, default=None, help="Optional fields.npz with df and angle")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--peer", type=Path, default=None, help="Second view, used only for the depth-profile figure")
    parser.add_argument("--depth0", type=Path, default=None)
    parser.add_argument("--depth1", type=Path, default=None)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    args = parser.parse_args()
    net = None
    device = None
    if args.fields is None or (args.peer is not None and args.depth0 is None):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = load_deeplsd(args.weight, device)
    bgr, distance, angle = _load_fields(args.image, args.fields, net, device)
    endpoints, samples = extract_ridge_lines(distance, angle)
    canvas = draw_ridge_lines(bgr, endpoints)
    out = args.out or args.image.with_name(args.image.stem + "_ridges.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    np.save(out.with_suffix(".npy"), endpoints)
    print(f"[+] {out}  lines {len(endpoints)}")
    if args.peer is not None and args.depth0 is not None and args.depth1 is not None:
        depth0 = cv2.imread(str(args.depth0), cv2.IMREAD_UNCHANGED)
        depth1 = cv2.imread(str(args.depth1), cv2.IMREAD_UNCHANGED)
        if depth0.ndim == 3:
            depth0 = depth0[..., 0]
        if depth1.ndim == 3:
            depth1 = depth1[..., 0]
        peer = cv2.imread(str(args.peer), cv2.IMREAD_COLOR)
        profile = out.with_name(out.stem + "_l2.jpg")
        _profile_figure(bgr, peer, samples, depth0.astype(np.float32), depth1.astype(np.float32), profile)
        print(f"[+] {profile}")


if __name__ == "__main__":
    main()
