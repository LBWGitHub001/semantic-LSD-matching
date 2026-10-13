"""Download a sample image and run DeepLabv3 inference."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .infer import main as infer_main
import sys


SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "demo.jpg"

    if not image_path.is_file():
        print(f"downloading sample -> {image_path}")
        urllib.request.urlretrieve(SAMPLE_URL, image_path)

    # Reuse CLI: python -m deeplab.demo  ==  infer on demo.jpg
    sys.argv = [
        "deeplab.infer",
        str(image_path),
        "-o",
        str(out_dir),
        "-b",
        "resnet50",
    ]
    infer_main()


if __name__ == "__main__":
    main()
