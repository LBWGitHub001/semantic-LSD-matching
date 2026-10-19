"""Download a sample image and run DeepLabv3 inference (project root entry)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deeplab.infer import main as infer_main

SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        image_path = Path(sys.argv[1])
    else:
        image_path = out_dir / "demo.jpg"
        if not image_path.is_file():
            print(f"downloading sample -> {image_path}")
            urllib.request.urlretrieve(SAMPLE_URL, image_path)

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
