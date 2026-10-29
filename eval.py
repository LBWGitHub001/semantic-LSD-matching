"""
Unified Evaluation and Benchmark Entrypoint for Object and Line Matching Models.
Implemented via the Extensible Factory Architecture in the `evaluation` package.

Usage:
  # 1. Evaluate any object matcher checkpoint (auto-detects architecture):
  python eval.py --checkpoint checkpoints/xfeat_object_matcher/best_model.pth --data-dir data/structured3d_semantic_regions --out-dir outputs/semantic_matching_vis --num-vis 50

  # 2. Evaluate latest training run automatically:
  python eval.py --checkpoint checkpoints/xfeat_object_matcher/latest --split test

  # 3. Call as a Python factory function:
  from eval import evaluate
  results = evaluate("checkpoints/xfeat_object_matcher/best_model.pth", data_dir="data/structured3d_semantic_regions")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# Re-export all evaluation components for full backward compatibility
from evaluation import (
    resolve_checkpoint_path,
    ModelFactory,
    DatasetFactory,
    BaseEvaluator,
    ObjectMatchingEvaluator,
    LineMatchingEvaluator,
    EvaluatorFactory,
    evaluate,
)

__all__ = [
    "resolve_checkpoint_path",
    "ModelFactory",
    "DatasetFactory",
    "BaseEvaluator",
    "ObjectMatchingEvaluator",
    "LineMatchingEvaluator",
    "EvaluatorFactory",
    "evaluate",
]


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Unified Evaluator for Line & Object Matchers")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint file or run directory (containing best_model.pth / last_model.pth / latest)")
    parser.add_argument("--data-dir", type=str, default="data/structured3d_semantic_regions",
                        help="Path to dataset directory (defaults to semantic regions)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--out-dir", type=str, default="outputs/semantic_matching_vis",
                        help="Directory to save diagnostic visualization images and reports")
    parser.add_argument("--num-vis", type=int, default=5,
                        help="Number of qualitative sample images to render (0 to disable)")
    parser.add_argument("--min-area-ratio", type=float, default=0.50,
                        help="Mutual co-visible area ratio threshold (default: 0.50)")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Optional model override (xfeat_object, xfeat_line_matcher, mask_line, line_transformer)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker count (default: 2)")
    return parser.parse_args(), parser


def main() -> None:
    args, _parser = parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        out_dir=args.out_dir,
        num_vis=args.num_vis,
        min_area_ratio=args.min_area_ratio,
        model_type=args.model_type,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
