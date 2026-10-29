#!/usr/bin/env python3
"""Evaluate a homo_matcher checkpoint: L1, L2, and warped ridges.

  python scripts/eval_homo_matcher.py \\
      --checkpoint checkpoints/homo_matcher/best_model.pth \\
      --data-dir data/homo_matcher
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluators import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DeepLSD homography: L1 cycle and L2 depth profiles.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "homo_matcher")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "homo_matcher_eval")
    parser.add_argument("--num-vis", type=int, default=5)
    args = parser.parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        out_dir=args.out_dir,
        num_vis=args.num_vis,
        model_type="FCN_field",
    )


if __name__ == "__main__":
    main()
