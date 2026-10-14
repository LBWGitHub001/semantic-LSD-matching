"""CLI: run DeepLabv3 semantic segmentation on an image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Support both: `python -m deeplab.infer` and running this file in an IDE.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deeplab.model import available_backbones, build_deeplabv3
    from deeplab.utils import (
        colorize_mask,
        load_image,
        overlay_mask,
        predict_mask,
        summarize_mask,
    )
else:
    from .model import available_backbones, build_deeplabv3
    from .utils import (
        colorize_mask,
        load_image,
        overlay_mask,
        predict_mask,
        summarize_mask,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepLabv3 semantic segmentation (torchvision pretrained)"
    )
    parser.add_argument("image", type=Path, help="Path to an input RGB image")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for mask / overlay images (default: outputs)",
    )
    parser.add_argument(
        "-b",
        "--backbone",
        choices=available_backbones(),
        default="resnet101",
        help="Backbone (default: resnet101)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay blend factor in [0, 1] (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"device={device}, backbone={args.backbone}")

    model, weights = build_deeplabv3(
        backbone=args.backbone, pretrained=True, device=device
    )
    image = load_image(args.image)
    mask = predict_mask(model, image, weights, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    mask_path = args.output_dir / f"{stem}_mask.png"
    overlay_path = args.output_dir / f"{stem}_overlay.png"

    colorize_mask(mask).save(mask_path)
    overlay_mask(image, mask, alpha=args.alpha).save(overlay_path)

    print("classes (name, pixels):")
    for name, count in summarize_mask(mask):
        print(f"  {name}: {count}")
    print(f"saved: {mask_path}")
    print(f"saved: {overlay_path}")


if __name__ == "__main__":
    main()
