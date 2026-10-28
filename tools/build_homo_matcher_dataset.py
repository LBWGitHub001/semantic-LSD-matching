#!/usr/bin/env python3
"""Build the DeepLSD homography dataset through DatasetGeneratorFactory.

Each pair stores two RGB images, two depth maps, and the two DeepLSD fields
(distance and angle). Line segments are not saved.

  python tools/build_homo_matcher_dataset.py --max-scenes 1 --max-pairs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_dataset import DatasetGeneratorFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Register and build the homo_matcher dataset.")
    parser.add_argument("--s3d-dir", type=Path, default=ROOT / "Structured3D_perspective_full_00" / "Structured3D")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "homo_matcher")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--weight", type=Path, default=None)
    args = parser.parse_args()
    factory = DatasetGeneratorFactory.create(
        task="homo_matcher",
        config={"s3d_dir": str(args.s3d_dir)},
    )
    component = factory.components[0]
    weight_kwargs = {"weight": args.weight} if args.weight else {}
    component.build_dataset(
        out_dir=args.out_dir,
        s3d_dir=args.s3d_dir,
        max_scenes=args.max_scenes,
        max_pairs=args.max_pairs,
        **weight_kwargs,
    )


if __name__ == "__main__":
    main()
