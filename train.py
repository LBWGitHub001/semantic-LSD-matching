"""
Unified Training Entrypoint & Facade for all Matching Networks.

This script acts purely as the external CLI entrypoint and argument analyzer.
Model-specific training routines are isolated within each respective package:
  - SuperPoint:               models.superpoint.train
  - XFeatObjectMatcher:       models.object_matcher.train
  - MaskLineFeatMatchNet:     models.mask_line_matcher.train

Usage:
  python train.py --config configs/superpoint.yaml
  python train.py --config configs/xfeat_object.yaml --epochs 20
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from configs import apply_config_to_parser, load_config, pre_parse_config_path
from factory_args import warn_unused_arguments
from models import model_family
from training_utils import (
    TQDM_BAR_FORMAT,
    get_git_commit,
    safe_load_checkpoint,
    save_config_json,
    update_latest_symlink,
    worker_init_fn,
)

# Re-export evaluate_object_pipeline for evaluation tests and backward compatibility
from models.object_matcher.eval import evaluate_object_pipeline

TRAIN_DISPATCH: dict[str, str] = {
    # 1. SuperPoint
    "superpoint": "models.superpoint.train",
    "SuperPoint": "models.superpoint.train",
    # 2. Object Matcher ("连连看") & Hybrid Line Matcher
    "xfeat_object": "models.object_matcher.train",
    "xfeat": "models.object_matcher.train",
    "object": "models.object_matcher.train",
    "xfeat_line_matcher": "models.object_matcher.train",
    "xfeat_line": "models.object_matcher.train",
    "line_object": "models.object_matcher.train",
    # 3. Continuous Geometry Field Line Matcher
    "mask_line": "models.mask_line_matcher.train",
    "mask": "models.mask_line_matcher.train",
}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Parse and validate unified command-line and YAML arguments."""
    parser = argparse.ArgumentParser(description="Unified Trainer for Line & Object Matching Networks")

    # Configuration file
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML configuration file (e.g., configs/superpoint.yaml)")

    # Model Architecture Selection
    parser.add_argument("--model", "--model-type", dest="model_type", type=str, default="xfeat_object",
                        help="Model architecture to train (xfeat_object, xfeat_line_matcher, mask_line, SuperPoint)")

    # Core Training Hyperparameters
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to training dataset directory")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Number of full training epochs (default: 12)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Base learning rate (model-dependent default)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay coefficient (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (currently fixed to 1 due to variable patch counts)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps before optimizer step (default: 1)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable PyTorch Automatic Mixed Precision (AMP FP16)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Hardware execution device ('cuda' or 'cpu')")

    # Checkpoint & Validation Scheduling
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save experiment checkpoints and logs")
    parser.add_argument("--val-interval", type=int, default=1,
                        help="Validate every N epochs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--save-last", action="store_true", default=True,
                        help="Always save last_model.pth after every epoch")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from an existing .pth checkpoint file")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Truncate training to N steps per epoch (useful for testing)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom experiment run name (overrides auto timestamp)")
    parser.add_argument("--label", type=str, default=None,
                        help="Optional tag appended to checkpoint directory name")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Disable timestamp in checkpoint directory")

    # DataLoader Settings
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker subprocess count (default: 2)")
    parser.add_argument("--data-ratio", type=float, default=1.0,
                        help="Random subset fraction of dataset to use for training (0.0 to 1.0, default: 1.0)")

    # Model Hyperparameters
    parser.add_argument("--d-model", type=int, default=128,
                        help="Transformer projection and feature dimension (default: 128)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of self/cross-attention Transformer layers (default: 2)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of multi-head attention heads (default: 4)")
    parser.add_argument("--use-pyramid", action="store_true", default=True,
                        help="Enable multi-scale FPN pyramid visual features (default: True)")

    # Object Matching Specific Arguments
    parser.add_argument("--dustbin-weight", type=float, default=0.5,
                        help="Loss weight for dustbin unmatched penalties (default: 0.5)")
    parser.add_argument("--motion-weight", type=float, default=0.2,
                        help="Loss weight for geometric spatial motion consistency (default: 0.2)")
    parser.add_argument("--motion-margin", type=float, default=0.1,
                        help="Margin threshold for motion distance loss (default: 0.1)")
    parser.add_argument("--min-area-ratio", type=float, default=0.50,
                        help="Minimum mutual co-visible area ratio (default: 0.50)")

    # Line Matching Specific Arguments
    parser.add_argument("--min-matches", type=int, default=2,
                        help="Minimum ground-truth line matches required per pair (default: 2)")

    # SuperPoint Stage & Loss Lambdas
    parser.add_argument("--train-stage", type=str, default="joint", choices=["joint", "detector", "descriptor"],
                        help="SuperPoint training stage: 'joint', 'detector', or 'descriptor'")
    parser.add_argument("--weight-desc", type=float, default=1.0, help="SuperPoint: descriptor loss weight (lambda_desc)")
    parser.add_argument("--positive-weight", type=float, default=250.0, help="SuperPoint: descriptor positive pair hinge weight")
    parser.add_argument("--positive-margin", type=float, default=1.0, help="SuperPoint: descriptor positive pair hinge margin")
    parser.add_argument("--negative-margin", type=float, default=0.2, help="SuperPoint: descriptor negative pair hinge margin")
    parser.add_argument("--weight-det", type=float, default=1.0, help="SuperPoint: detector loss weight (lambda_det)")
    parser.add_argument("--det-pos-weight", type=float, default=50.0, help="SuperPoint: detector positive keypoint cell weight")
    parser.add_argument("--weight-orphan", type=float, default=1.0, help="SuperPoint: orphan endpoint penalty weight (lambda_orphan)")
    parser.add_argument("--weight-len", type=float, default=0.5, help="SuperPoint: max line length reward weight (lambda_len)")
    parser.add_argument("--length-exponent", type=float, default=1.5, help="SuperPoint: super-linear max line length reward exponent")
    parser.add_argument("--weight-premature", type=float, default=0.5, help="SuperPoint: premature ray probing penalty weight (lambda_premature)")
    parser.add_argument("--probe-dist", type=float, default=4.0, help="SuperPoint: premature ray probing extrapolation distance in px")
    parser.add_argument("--tau-df", type=float, default=0.55, help="SuperPoint: distance field line ridge threshold")
    parser.add_argument("--tau-ang", type=float, default=0.10, help="SuperPoint: angle field tangent alignment threshold")
    parser.add_argument("--min-length", type=float, default=8.0, help="SuperPoint: minimum valid line length in pixels")
    parser.add_argument("--num-candidates", type=int, default=16, help="SuperPoint: number of candidate endpoints extracted")
    parser.add_argument("--num-line-samples", type=int, default=10, help="SuperPoint: number of interpolation samples along line")

    cfg_path = pre_parse_config_path(argv)
    if cfg_path:
        cfg_dict = load_config(cfg_path)
        apply_config_to_parser(parser, cfg_dict, section="train")

    return parser.parse_args(argv), parser


def train_applicable_arguments(model_type: str) -> set[str]:
    """Destinations this training entrypoint validates for the given model architecture."""
    common = {
        "model_type", "data_dir", "epochs", "lr", "weight_decay",
        "d_model", "num_layers", "save_dir", "run_name", "label",
        "no_timestamp", "device", "max_steps",
    }
    family = model_family(model_type)
    if family == "point":
        return (common - {"d_model", "num_layers"}) | {
            "train_stage", "weight_desc", "positive_weight", "positive_margin", "negative_margin",
            "weight_det", "det_pos_weight", "weight_orphan", "weight_len", "length_exponent",
            "weight_premature", "probe_dist", "tau_df", "tau_ang", "min_length", "num_candidates", "num_line_samples",
        }
    if family == "object":
        return common | {
            "nhead", "grad_accum", "num_workers", "amp", "val_interval",
            "save_interval", "save_last", "resume", "dustbin_weight",
            "motion_weight", "motion_margin", "min_area_ratio",
            "data_ratio", "use_pyramid",
        }
    return common | {
        "min_matches", "num_workers", "amp", "val_interval",
        "save_interval", "save_last", "resume", "data_ratio",
    }


def dispatch_train(args: argparse.Namespace) -> Path:
    """Dispatches execution to the self-contained training module for the selected model."""
    target_module_str = TRAIN_DISPATCH.get(args.model_type)
    if not target_module_str:
        family = model_family(args.model_type)
        if family == "point":
            target_module_str = "models.superpoint.train"
        elif family == "object":
            target_module_str = "models.object_matcher.train"
        else:
            target_module_str = "models.mask_line_matcher.train"

    train_module = importlib.import_module(target_module_str)
    return train_module.train(args)


def main(argv: list[str] | None = None) -> Path:
    args, parser = parse_args(argv)
    warn_unused_arguments(parser, train_applicable_arguments(args.model_type), args.model_type, argv)
    return dispatch_train(args)


if __name__ == "__main__":
    main()
