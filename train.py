"""
Unified Training Entrypoint for Object and Line Matching Networks.

Usage:
  # 1. Train Object Matcher ('连连看' XFeatObjectMatcher):
  python train.py --model xfeat_object --epochs 12 --lr 5e-4 --amp

  # 2. Train 6-Channel Geometry Field Line Matcher (MaskLineFeatMatchNet):
  python train.py --model mask_line --epochs 8 --lr 2e-4 --amp

  # 3. Train Baseline Line-Transformer (LineMatcherNet):
  python train.py --model line_transformer --epochs 8 --lr 2e-4

  # 4. Train descriptor-only SuperPoint on planar letters:
  python train.py --model SuperPoint
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback generator if tqdm is missing
    def tqdm(iterable, *args, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluators import build_object_forward_kwargs
from models import (
    build_loss,
    build_model,
    get_available_models,
    is_semantic_line_model,
    model_family,
    uses_patch_lines,
)
from datasets import (
    ObjectMatchingDataset,
    collate_single_pair,
    predict_bipartite_matches,
    LineFeatPairDataset,
)
from visualization.matching_visualizer import save_pair_visualization
from tools.experiment_tracker import ExperimentLogger


def safe_load_checkpoint(path: Path | str, map_location: Any = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_git_commit() -> str:
    """Retrieves short git commit hash if available."""
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def save_config_json(config_path: Path, config_data: dict) -> None:
    """Atomically writes an explanatory configuration and experiment metadata JSON and YAML."""
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        from configs import save_config
        save_config(config_data, config_path.with_suffix(".yaml"))
    except Exception as e:
        print(f"[!] Warning: Failed to write {config_path.name}: {e}")


def update_latest_symlink(base_dir: Path, target_dir: Path) -> None:
    """Updates a 'latest' symlink in base_dir pointing to target_dir."""
    try:
        latest_link = base_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(target_dir.name, target_is_directory=True)
    except Exception:
        pass


# Fixed-width progress bar format to completely eliminate visual jitter/jumping ("鬼畜")
TQDM_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def worker_init_fn(worker_id: int) -> None:
    """Disables OpenCV internal multi-threading within worker processes to prevent thread contention."""
    cv2.setNumThreads(0)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Unified Trainer for Line & Object Matching Networks")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML configuration file (e.g., configs/superpoint.yaml)")
    parser.add_argument("--model", "--model-type", dest="model_type", type=str, default="xfeat_object",
                        choices=get_available_models(),
                        help="Model architecture: 'xfeat_object', 'xfeat_line_matcher', 'semantic_line_matcher', 'mask_line', 'line_transformer'")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory or index path")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 5e-4 for xfeat_object, 1e-4 for line models)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--min-matches", type=int, default=3, help="Minimum GT matches per pair (for line models)")
    parser.add_argument("--d-model", type=int, default=128, help="Feature dimension (default: 128)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes (default: 4)")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (FP16)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Checkpoint directory")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run subfolder name (default: start time YYYYMMDD_HHMMSS)")
    parser.add_argument("--label", type=str, default=None, help="Optional user-defined custom tag/label to annotate this training experiment")
    parser.add_argument("--no-timestamp", action="store_true", default=False, help="Save checkpoints directly in save-dir without timestamp subfolder")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval in epochs")
    parser.add_argument("--save-interval", type=int, default=5, help="Save historical checkpoint every N epochs (e.g. epoch_005.pth, default: 5)")
    parser.add_argument("--no-save-last", dest="save_last", action="store_false", default=True, help="Disable saving last_model.pth")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    parser.add_argument("--dustbin-weight", type=float, default=0.3, help="Dustbin rejection loss weight for ObjectMatchingLoss (default: 0.3)")
    parser.add_argument("--motion-weight", type=float, default=0.2, help="Transformation consistency loss weight (方案一: 位移场共识损失, default: 0.2)")
    parser.add_argument("--motion-margin", type=float, default=0.15, help="Parallax tolerance margin in [-1, 1] normalized coordinates (default: 0.15)")
    parser.add_argument("--region-weight", type=float, default=1.0, help="semantic_line_matcher: region KL weight")
    parser.add_argument("--line-weight", type=float, default=1.0, help="semantic_line_matcher: line KL weight")
    parser.add_argument("--sem-weight", type=float, default=0.1, help="semantic_line_matcher: line→semantic JSD weight")
    parser.add_argument("--geo-weight", type=float, default=0.2, help="semantic_line_matcher: adjacency geo weight")
    parser.add_argument("--adj-sigma", type=float, default=0.3, help="semantic_line_matcher: soft adjacency temperature")
    parser.add_argument("--line-sample-points", type=int, default=8, help="semantic_line_matcher: along-line sample count")
    parser.add_argument("--cycle-weight", type=float, default=1.0, help="FCN_field: weight of the velocity cycle")
    parser.add_argument("--depth-weight", type=float, default=1.0, help="FCN_field: weight of the ridge depth loss")
    parser.add_argument("--flow-weight", type=float, default=1.0, help="FCN_field: weight of the geometric velocity loss")
    parser.add_argument("--min-area-ratio", type=float, default=0.50, help="Minimum mutually matched region area ratio to allow matching (default: 0.50)")
    parser.add_argument("--train-stage", type=str, default="joint", choices=["joint", "descriptor", "detector"],
                        help="Training stage for SuperPoint: 'joint' (default), 'descriptor' (freeze detector), or 'detector' (freeze descriptor)")
    parser.add_argument("--data-ratio", "--data-fraction", dest="data_ratio", type=float, default=1.0, help="Fraction of training dataset to randomly sample (0.0 to 1.0, default: 1.0)")
    parser.add_argument("--pyramid", dest="use_pyramid", action="store_true", default=True, help="Enable multi-scale feature pyramid in ObjectMatcher (default: True)")
    parser.add_argument("--no-pyramid", dest="use_pyramid", action="store_false", help="Disable multi-scale feature pyramid (use single-scale legacy backbone)")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per epoch for quick dry-run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # SuperPoint specific loss weights & geometric hyperparameters
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

    from configs import load_config, apply_config_to_parser, pre_parse_config_path

    cfg_path = pre_parse_config_path(argv)
    if cfg_path:
        cfg_dict = load_config(cfg_path)
        apply_config_to_parser(parser, cfg_dict, section="train")

    return parser.parse_args(argv), parser


def train_applicable_arguments(model_type: str) -> set[str]:
    """Destinations this training factory actually reads."""
    from models import is_semantic_line_model, model_family

    common = {
        "model_type", "data_dir", "epochs", "lr", "weight_decay",
        "d_model", "num_layers", "save_dir", "run_name", "label",
        "no_timestamp", "device", "max_steps",
    }
    family = model_family(model_type)
    if family == "homography":
        return (common - {"d_model", "num_layers"}) | {"line_sample_points"}
    if family == "point":
        return (common - {"d_model", "num_layers"}) | {
            "train_stage", "weight_desc", "positive_weight", "positive_margin", "negative_margin",
            "weight_det", "det_pos_weight", "weight_orphan", "weight_len", "length_exponent",
            "weight_premature", "probe_dist", "tau_df", "tau_ang", "min_length", "num_candidates", "num_line_samples",
        }
    if family == "object" and is_semantic_line_model(model_type):
        return common | {
            "nhead", "grad_accum", "num_workers", "amp", "val_interval",
            "save_interval", "save_last", "resume", "dustbin_weight",
            "min_area_ratio", "data_ratio", "use_pyramid",
            "region_weight", "line_weight", "sem_weight", "geo_weight",
            "adj_sigma", "line_sample_points",
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


# ==============================================================================
# Object Matching Training Pipeline
# ==============================================================================
def train_object_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir or "data/synthetic_object_matching_scenes")
    default_ckpt_dir = "checkpoints/xfeat_object_matcher" if args.model_type in ["xfeat_object", "xfeat", "object"] else f"checkpoints/{args.model_type}"
    base_save_dir = Path(args.save_dir or default_ckpt_dir)
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d_%H%M%S")
    if args.no_timestamp:
        save_dir = base_save_dir
        run_name = base_save_dir.name
    else:
        if args.run_name:
            run_name = args.run_name
        else:
            pyr_tag = "pyramid" if getattr(args, "use_pyramid", True) else "no_pyramid"
            d_tag = f"d{getattr(args, 'd_model', 128)}"
            dataset_name = Path(data_dir).name
            label_tag = f"+【{args.label}】" if getattr(args, "label", None) else ""
            run_name = f"{args.model_type}+{pyr_tag}_{d_tag}+{dataset_name}{label_tag}+【{start_time_str}】"
        save_dir = base_save_dir / run_name

    save_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(base_save_dir, save_dir)
    config_json_path = save_dir / "config.json"

    lr = args.lr if args.lr is not None else 5e-4
    device = torch.device(args.device)

    print("=================================================================")
    print("      Unified Trainer: XFeatObjectMatcher ('连连看') Training     ")
    print("=================================================================")
    print(f"  Device:             {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Dataset Directory:  {data_dir}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Learning Rate:      {lr}")
    print(f"  Dustbin Weight:     {args.dustbin_weight}")
    print(f"  Motion Loss Weight: {args.motion_weight} (Margin: {args.motion_margin})")
    print(f"  Min Area Ratio:     {args.min_area_ratio * 100:.0f}% (Mutual Co-visible Region Filter)")
    if getattr(args, "data_ratio", 1.0) < 1.0:
        print(f"  Data Ratio:         {args.data_ratio * 100:.1f}% (Random Subset Fraction)")
    else:
        print(f"  Data Ratio:         100% (Full Dataset)")
    print(f"  Pyramid Features:   {'Enabled (Multi-Scale FPN)' if getattr(args, 'use_pyramid', True) else 'Disabled (Single-Scale)'}")
    print(f"  Model Dimension:    {args.d_model} (Layers: {args.num_layers}, Heads: {args.nhead})")
    print(f"  Mixed Precision:    {'Enabled (AMP FP16)' if args.amp else 'Disabled (FP32)'}")
    print(f"  Checkpoint Base:    {base_save_dir}")
    print(f"  Run Directory:      {save_dir} (Run: {run_name})")
    print(f"  Save Interval:      Every {args.save_interval} epoch(s)")
    if args.resume:
        print(f"  Resume From:        {args.resume}")
    print("=================================================================\n")

    # Auto-detect whether data_dir is a SemanticRegionDataset or ObjectMatchingDataset
    use_region_ds = False
    p_data = Path(data_dir)
    if (p_data / "train_split.json").exists():
        with open(p_data / "train_split.json", "r", encoding="utf-8") as f:
            splits_data = json.load(f)
            if splits_data and (p_data / splits_data[0] / "correspondence.json").exists():
                use_region_ds = True

    is_semantic_line = is_semantic_line_model(args.model_type)
    if is_semantic_line:
        # Dual line+region matcher: disable legacy global motion by default
        if args.motion_weight == 0.2:
            args.motion_weight = 0.0
    needs_lines = uses_patch_lines(args.model_type)
    if use_region_ds:
        from datasets.semantic_region_dataset import SemanticRegionDataset, collate_region_pairs
        train_ds = SemanticRegionDataset(dataset_dir=data_dir, split="train", min_area_ratio=args.min_area_ratio, load_lines=needs_lines)
        val_ds = SemanticRegionDataset(dataset_dir=data_dir, split="val", min_area_ratio=args.min_area_ratio, load_lines=needs_lines)
        test_ds = SemanticRegionDataset(dataset_dir=data_dir, split="test", min_area_ratio=args.min_area_ratio, load_lines=needs_lines)
        collate_fn = collate_region_pairs
    else:
        train_ds = ObjectMatchingDataset(data_dir=data_dir, split="train")
        val_ds = ObjectMatchingDataset(data_dir=data_dir, split="val")
        test_ds = ObjectMatchingDataset(data_dir=data_dir, split="test")
        collate_fn = collate_single_pair

    orig_train_len = len(train_ds)
    if getattr(args, "data_ratio", 1.0) < 1.0:
        ratio = max(0.001, min(1.0, float(args.data_ratio)))
        subset_size = max(1, int(orig_train_len * ratio))
        rng = np.random.RandomState(42)  # reproducible random subset
        subset_indices = rng.choice(orig_train_len, size=subset_size, replace=False).tolist()
        train_ds = torch.utils.data.Subset(train_ds, subset_indices)
        print(f"[*] Dataset subset: Randomly sampled {len(train_ds)}/{orig_train_len} ({ratio*100:.1f}%) pairs for training.\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Evaluate sequentially in main process to eliminate 5.8GB worker memory leak
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model_kwargs = dict(
        feat_dim=64,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        sinkhorn_iters=20,
        use_pyramid=getattr(args, "use_pyramid", True),
    )
    if is_semantic_line:
        model_kwargs.update(
            with_geometry=True,
            line_sample_points=getattr(args, "line_sample_points", 8),
            enable_line_head=True,
            line_dim=64,
        )
    model = build_model(args.model_type, **model_kwargs).to(device)

    if is_semantic_line:
        loss_fn = build_loss(
            args.model_type,
            dustbin_weight=args.dustbin_weight,
            region_weight=args.region_weight,
            line_weight=args.line_weight,
            sem_weight=args.sem_weight,
            geo_weight=args.geo_weight,
            adj_sigma=args.adj_sigma,
            motion_weight=0.0,
        ).to(device)
    else:
        loss_fn = build_loss(
            args.model_type,
            dustbin_weight=args.dustbin_weight,
            motion_weight=args.motion_weight,
            motion_margin=args.motion_margin,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=2048.0)

    start_epoch = 1
    best_val_f1 = 0.0
    best_ckpt_path = save_dir / "best_model.pth"
    last_ckpt_path = save_dir / "last_model.pth"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume not found: {resume_path}")
        print(f"[*] Resuming object training from checkpoint: {resume_path}")
        ckpt = safe_load_checkpoint(resume_path, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if args.lr is not None:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
            except Exception as e:
                print(f"    [!] Warning: Failed to load optimizer state: {e}")
        if args.amp and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print(f"    [!] Warning: Failed to load scaler state: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("best_val_f1", ckpt.get("val_metrics", {}).get("f1", 0.0))
        print(f"    Resumed at epoch {start_epoch}, previous best Val F1: {best_val_f1:.2f}%\n")

    num_total_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config_data = {
        "experiment_info": {
            "run_name": run_name,
            "label": getattr(args, "label", None),
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "git_commit": get_git_commit(),
            "command": " ".join(sys.argv),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "model": {
            "model_type": args.model_type,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "nhead": args.nhead,
            "use_pyramid": getattr(args, "use_pyramid", True),
            "sinkhorn_iters": 20,
            "total_parameters": num_total_params,
            "trainable_parameters": num_trainable_params,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": lr,
            "weight_decay": args.weight_decay,
            "grad_accum": args.grad_accum,
            "amp": args.amp,
            "dustbin_weight": args.dustbin_weight,
            "motion_loss_weight": args.motion_weight,
            "motion_margin": args.motion_margin,
            "min_area_ratio": args.min_area_ratio,
            "data_ratio": getattr(args, "data_ratio", 1.0),
            "save_interval": args.save_interval,
            "save_last": args.save_last,
            "val_interval": args.val_interval,
            "num_workers": args.num_workers,
            "all_cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        "dataset": {
            "dataset_type": "SemanticRegionDataset" if use_region_ds else "ObjectMatchingDataset",
            "dataset_dir": str(p_data.resolve()),
            "total_train_pairs": orig_train_len,
            "train_pairs": len(train_ds),
            "data_ratio": getattr(args, "data_ratio", 1.0),
            "val_pairs": len(val_ds),
            "test_pairs": len(test_ds),
        },
        "checkpoints": {
            "run_dir": str(save_dir.resolve()),
            "best_checkpoint": None,
            "last_checkpoint": "last_model.pth" if args.save_last else None,
            "historical_checkpoints": [],
        },
        "best_validation": None,
        "test_benchmark": None,
    }
    save_config_json(config_json_path, config_data)
    logger = ExperimentLogger(save_dir=save_dir, config_data=config_data, enable_stdout_tee=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        t0 = time.time()

        optimizer.zero_grad()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]",
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )

        for step, batch in enumerate(pbar):
            if args.max_steps is not None and step >= args.max_steps:
                break
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)
            model_kwargs = build_object_forward_kwargs(batch, device)

            c0 = batch["centroids0"].to(device) if "centroids0" in batch and batch["centroids0"] is not None else None
            c1 = batch["centroids1"].to(device) if "centroids1" in batch and batch["centroids1"] is not None else None

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(patches0, patches1, **model_kwargs)
                if is_semantic_line:
                    log_line = out.get("log_P_line")
                    if torch.is_tensor(log_line):
                        log_line = log_line.float()
                    loss_out = loss_fn(
                        log_P_region=out["log_P"].float(),
                        log_P_line=log_line,
                        gt_matches=batch["gt_matches"],
                        unmatched_0=batch["unmatched_0"],
                        unmatched_1=batch["unmatched_1"],
                        gt_line_pairs_flat=batch.get("gt_line_pairs_flat", [[]]),
                        unmatched_lines_flat_0=batch.get("unmatched_lines_flat_0", [[]]),
                        unmatched_lines_flat_1=batch.get("unmatched_lines_flat_1", [[]]),
                        line_parents0=out.get("line_parents0"),
                        line_parents1=out.get("line_parents1"),
                        centroids0=c0,
                        centroids1=c1,
                        has_line_ids=bool(batch.get("has_line_ids", False)),
                    )
                    loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
                else:
                    loss = loss_fn(
                        out["log_P"].float(),
                        batch["gt_matches"],
                        batch["unmatched_0"],
                        batch["unmatched_1"],
                        centroids0=c0,
                        centroids1=c1,
                    )
                loss_scaled = loss.float() / args.grad_accum

            scaler.scale(loss_scaled).backward()
            step_count += 1

            if step_count % args.grad_accum == 0 or step == len(train_loader) - 1:
                has_grads = any(p.grad is not None for p in model.parameters())
                if args.amp:
                    if has_grads:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    if has_grads:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                optimizer.zero_grad()

            loss_val = float(loss.item())
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
            cur_avg_loss = total_loss / max(step_count, 1)

            # Real-time status in progress bar (fixed-width formatting to avoid bar jitter)
            m0_num = patches0.shape[1]
            m1_num = patches1.shape[1]
            n_gt = len(batch["gt_matches"][0]) if "gt_matches" in batch and batch["gt_matches"] else 0
            mem_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            pbar.set_postfix_str(
                f"loss={loss_val:6.4f}, avg={cur_avg_loss:6.4f}, patch={m0_num:>2d}v{m1_num:<2d}, gt={n_gt:>2d}, gpu={mem_mb:>4.0f}MB"
            )

        train_loss = total_loss / max(step_count, 1)
        epoch_sec = time.time() - t0

        val_metrics = None
        is_best = False
        if epoch % args.val_interval == 0:
            val_metrics = evaluate_object_pipeline(
                model, val_loader, device,
                desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]",
                min_area_ratio=args.min_area_ratio,
            )
            val_f1 = val_metrics["f1"]
            is_best = val_f1 >= best_val_f1
            if is_best:
                best_val_f1 = val_f1

            star = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) "
                  f"| Loss: {train_loss:.4f} "
                  f"| Val Prec: {val_metrics['precision']:5.1f}% | Rec: {val_metrics['recall']:5.1f}% "
                  f"| F1: {val_f1:5.1f}% | Dustbin: P={val_metrics['dustbin_prec']:4.1f}%/R={val_metrics['dustbin_rec']:4.1f}%{star}")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | Loss: {train_loss:.4f}")

        # Log epoch metrics to CSV, memory history, and update training curves
        cur_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else lr
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_metrics=val_metrics,
            is_best=is_best,
            epoch_sec=epoch_sec,
            lr=cur_lr,
        )

        # Checkpoint Management
        serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if args.amp else None,
            "best_val_f1": best_val_f1,
            "val_metrics": val_metrics,
            "train_loss": train_loss,
            "args": serializable_args,
        }

        # 1. Update best checkpoint
        if is_best:
            torch.save(ckpt_data, best_ckpt_path)
            config_data["checkpoints"]["best_checkpoint"] = "best_model.pth"
            config_data["best_validation"] = {
                "epoch": epoch,
                "f1": round(val_f1, 2),
                "precision": round(val_metrics["precision"], 2),
                "recall": round(val_metrics["recall"], 2),
                "dustbin_precision": round(val_metrics["dustbin_prec"], 2),
                "dustbin_recall": round(val_metrics["dustbin_rec"], 2),
                "dustbin_accuracy": round(val_metrics["dustbin_acc"], 2),
                "cross_cat_fp": val_metrics["cross_cat_fp"],
            }
            save_config_json(config_json_path, config_data)

        # 2. Update latest checkpoint
        if args.save_last:
            torch.save(ckpt_data, last_ckpt_path)

        # 3. Save periodic historical checkpoint
        if args.save_interval > 0 and (epoch % args.save_interval == 0 or epoch == args.epochs):
            epoch_ckpt_path = save_dir / f"epoch_{epoch:03d}.pth"
            torch.save(ckpt_data, epoch_ckpt_path)
            print(f"  [Checkpoint] Saved historical checkpoint -> {epoch_ckpt_path.name}")
            if epoch_ckpt_path.name not in config_data["checkpoints"]["historical_checkpoints"]:
                config_data["checkpoints"]["historical_checkpoints"].append(epoch_ckpt_path.name)
                save_config_json(config_json_path, config_data)

        # Garbage collection & CUDA cache reclamation at epoch boundaries
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 80)
    print(f"[+] Training Complete! Best Val F1: {best_val_f1:.2f}%.")
    eval_ckpt_path = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
    if not eval_ckpt_path.exists():
        print("[!] No checkpoint found to evaluate.")
        return

    # Benchmark test set & save visualizations
    print(f"\nEvaluating checkpoint on unseen test scenes ({eval_ckpt_path.name})...")
    ckpt = safe_load_checkpoint(eval_ckpt_path, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)

    # 1. Unconstrained visual matching evaluation
    raw_test_metrics = evaluate_object_pipeline(
        model, test_loader, device, desc="Testing (Raw Visual)", min_area_ratio=args.min_area_ratio
    )

    # 2. Category-constrained evaluation (if class labels available)
    cat_test_metrics = evaluate_object_pipeline(
        model, test_loader, device, desc="Testing (Semantic Constrained)", enforce_same_category=True, min_area_ratio=args.min_area_ratio
    )

    print("-" * 75)
    print(f"TEST SET BENCHMARK RESULTS ({len(test_ds)} pairs):")
    print("  [1] Raw Visual Appearance Matching (Zero Class Prior):")
    print(f"      Precision          : {raw_test_metrics['precision']:.2f}%")
    print(f"      Recall             : {raw_test_metrics['recall']:.2f}%")
    print(f"      F1-Score           : {raw_test_metrics['f1']:.2f}%")
    print(f"      Matches            : {raw_test_metrics['tp']} TP, {raw_test_metrics['fp']} FP, {raw_test_metrics['fn']} FN")
    print(f"      Cross-Category FP  : {raw_test_metrics['cross_cat_fp']} ({raw_test_metrics['cross_cat_rate']:.1f}% of all FPs)")
    print(f"      Dustbin Precision  : {raw_test_metrics['dustbin_prec']:.2f}% (Recall: {raw_test_metrics['dustbin_rec']:.2f}%)")
    print("  [2] Semantic Category Constrained Matching:")
    print(f"      Precision          : {cat_test_metrics['precision']:.2f}%")
    print(f"      Recall             : {cat_test_metrics['recall']:.2f}%")
    print(f"      F1-Score           : {cat_test_metrics['f1']:.2f}%")
    print(f"      Matches            : {cat_test_metrics['tp']} TP, {cat_test_metrics['fp']} FP, {cat_test_metrics['fn']} FN")
    print("-" * 75)

    vis_dir = save_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    global_vis_dir = Path("outputs/semantic_matching_vis")
    global_vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for vis_idx, batch in enumerate(test_loader):
            if vis_idx >= 5:
                break
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)
            out = model(patches0, patches1, **build_object_forward_kwargs(batch, device))
            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(
                out["prob_matrix"][0],
                areas_0=batch.get("areas0"),
                areas_1=batch.get("areas1"),
                min_area_ratio=args.min_area_ratio,
            )
            scene_tag = batch.get("scene_name", f"pair_{vis_idx+1}")
            save_file = vis_dir / f"test_sample_{vis_idx + 1}_{scene_tag}.jpg"
            global_file = global_vis_dir / f"test_sample_{vis_idx + 1}_{scene_tag}.jpg"

            import shutil
            wrote = save_pair_visualization(
                batch,
                out,
                save_file,
                title=f"Sample {vis_idx + 1}: {scene_tag}",
                pred_matches=pred_matches,
                pred_un0=pred_un0,
                pred_un1=pred_un1,
                data_dir=p_data,
                scene_name=scene_tag,
                use_region_ds=use_region_ds,
            )
            if wrote:
                shutil.copyfile(save_file, global_file)

    config_data["experiment_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_data["test_benchmark"] = {
        "raw_visual": {
            "precision": round(raw_test_metrics["precision"], 2),
            "recall": round(raw_test_metrics["recall"], 2),
            "f1": round(raw_test_metrics["f1"], 2),
            "tp": raw_test_metrics["tp"],
            "fp": raw_test_metrics["fp"],
            "fn": raw_test_metrics["fn"],
            "cross_cat_fp": raw_test_metrics["cross_cat_fp"],
            "cross_cat_rate": round(raw_test_metrics["cross_cat_rate"], 2),
            "dustbin_precision": round(raw_test_metrics["dustbin_prec"], 2),
            "dustbin_recall": round(raw_test_metrics["dustbin_rec"], 2),
        },
        "semantic_constrained": {
            "precision": round(cat_test_metrics["precision"], 2),
            "recall": round(cat_test_metrics["recall"], 2),
            "f1": round(cat_test_metrics["f1"], 2),
            "tp": cat_test_metrics["tp"],
            "fp": cat_test_metrics["fp"],
            "fn": cat_test_metrics["fn"],
        },
    }
    save_config_json(config_json_path, config_data)
    logger.finish(
        raw_test_metrics=raw_test_metrics,
        cat_test_metrics=cat_test_metrics,
    )
    print(f"\n[+] Saved run configuration and benchmark summary to: {config_json_path}")


def evaluate_object_pipeline(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
    enforce_same_category: bool = False,
    min_area_ratio: float = 0.0,
) -> dict[str, float]:
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    cross_cat_fp = 0
    same_cat_fp = 0
    dustbin_tp, dustbin_fp, dustbin_fn, dustbin_tn = 0, 0, 0, 0

    pbar = tqdm(loader, desc=desc, bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair")
    with torch.no_grad():
        for batch in pbar:
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)
            out = model(patches0, patches1, **build_object_forward_kwargs(batch, device))

            cats0 = [o.get("category", "") for o in batch.get("objs0", [])] if enforce_same_category else None
            cats1 = [o.get("category", "") for o in batch.get("objs1", [])] if enforce_same_category else None

            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(
                out["prob_matrix"][0],
                thresh=0.2,
                categories_0=cats0,
                categories_1=cats1,
                areas_0=batch.get("areas0"),
                areas_1=batch.get("areas1"),
                min_area_ratio=min_area_ratio,
            )

            gt_matches = set(batch["gt_matches"][0])
            pred_set = set(pred_matches)

            cur_tp = len(pred_set.intersection(gt_matches))
            cur_fp_set = pred_set - gt_matches
            cur_fn = len(gt_matches - pred_set)

            total_tp += cur_tp
            total_fp += len(cur_fp_set)
            total_fn += cur_fn

            objs0 = batch.get("objs0", [])
            objs1 = batch.get("objs1", [])
            for i, j in cur_fp_set:
                c0 = objs0[i].get("category", "") if i < len(objs0) else ""
                c1 = objs1[j].get("category", "") if j < len(objs1) else ""
                if c0 and c1 and c0 != c1:
                    cross_cat_fp += 1
                else:
                    same_cat_fp += 1

            # Dustbin 4-class confusion matrix
            gt_un0 = set(batch["unmatched_0"][0])
            gt_un1 = set(batch["unmatched_1"][0])
            pred_un0_set = set(pred_un0)
            pred_un1_set = set(pred_un1)

            m0_num = patches0.shape[1]
            m1_num = patches1.shape[1]

            for i in range(m0_num):
                is_un_gt = (i in gt_un0)
                is_un_pred = (i in pred_un0_set)
                if is_un_gt and is_un_pred:
                    dustbin_tp += 1
                elif not is_un_gt and is_un_pred:
                    dustbin_fp += 1
                elif is_un_gt and not is_un_pred:
                    dustbin_fn += 1
                else:
                    dustbin_tn += 1

            for j in range(m1_num):
                is_un_gt = (j in gt_un1)
                is_un_pred = (j in pred_un1_set)
                if is_un_gt and is_un_pred:
                    dustbin_tp += 1
                elif not is_un_gt and is_un_pred:
                    dustbin_fp += 1
                elif is_un_gt and not is_un_pred:
                    dustbin_fn += 1
                else:
                    dustbin_tn += 1

            cur_p = total_tp / (total_tp + total_fp + 1e-6)
            cur_r = total_tp / (total_tp + total_fn + 1e-6)
            cur_f1 = 2.0 * cur_p * cur_r / (cur_p + cur_r + 1e-6)
            pbar.set_postfix_str(f"F1={cur_f1*100:5.1f}%, TP={total_tp:>3d}, FP={total_fp:>3d}, FN={total_fn:>3d}")

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    dustbin_prec = dustbin_tp / (dustbin_tp + dustbin_fp + 1e-6)
    dustbin_rec = dustbin_tp / (dustbin_tp + dustbin_fn + 1e-6)
    dustbin_acc = (dustbin_tp + dustbin_tn) / (dustbin_tp + dustbin_fp + dustbin_fn + dustbin_tn + 1e-6)
    cross_cat_rate = cross_cat_fp / (total_fp + 1e-6)

    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "dustbin_prec": dustbin_prec * 100.0,
        "dustbin_rec": dustbin_rec * 100.0,
        "dustbin_acc": dustbin_acc * 100.0,
        "cross_cat_fp": cross_cat_fp,
        "cross_cat_rate": cross_cat_rate * 100.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


# ==============================================================================
# Line Matching Training Pipeline
# ==============================================================================
def train_line_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir or "data/line_gt_dataset")
    base_save_dir = Path(args.save_dir or "checkpoints/linefeat")
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d_%H%M%S")
    if args.no_timestamp:
        save_dir = base_save_dir
        run_name = base_save_dir.name
    else:
        if args.run_name:
            run_name = args.run_name
        else:
            pyr_tag = "pyramid" if getattr(args, "use_pyramid", True) else "no_pyramid"
            d_tag = f"d{getattr(args, 'd_model', 128)}"
            dataset_name = Path(data_dir).name
            label_tag = f"+【{args.label}】" if getattr(args, "label", None) else ""
            run_name = f"{args.model_type}+{pyr_tag}_{d_tag}+{dataset_name}{label_tag}+【{start_time_str}】"
        save_dir = base_save_dir / run_name

    save_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(base_save_dir, save_dir)
    config_json_path = save_dir / "config.json"

    lr = args.lr if args.lr is not None else 1e-4
    device = torch.device(args.device)

    print("=================================================================")
    print(f"      Unified Trainer: Line Matching ({args.model_type})         ")
    print("=================================================================")
    print(f"  Device:           {device}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Learning Rate:    {lr}")
    print(f"  Checkpoint Base:  {base_save_dir}")
    print(f"  Run Directory:    {save_dir} (Run: {run_name})")
    print(f"  Save Interval:    Every {args.save_interval} epoch(s)")
    if args.resume:
        print(f"  Resume From:      {args.resume}")
    print("=================================================================\n")

    train_dataset = LineFeatPairDataset(
        index_path="data/line_gt_dataset/index.json",
        split="train",
        min_matches=args.min_matches,
        load_images=True,
    )
    val_dataset = LineFeatPairDataset(
        index_path="data/line_gt_dataset/index.json",
        split="val",
        min_matches=args.min_matches,
        load_images=True,
    )
    orig_line_train_len = len(train_dataset)
    if getattr(args, "data_ratio", 1.0) < 1.0:
        ratio = max(0.001, min(1.0, float(args.data_ratio)))
        subset_size = max(1, int(orig_line_train_len * ratio))
        rng = np.random.RandomState(42)
        subset_indices = rng.choice(orig_line_train_len, size=subset_size, replace=False).tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
        print(f"[*] Dataset subset: Randomly sampled {len(train_dataset)}/{orig_line_train_len} ({ratio*100:.1f}%) pairs for training.\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = build_model(
        args.model_type,
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)

    criterion = build_loss(args.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=2048.0)

    start_epoch = 1
    best_val_acc = 0.0
    best_ckpt_path = save_dir / "best_model.pth"
    last_ckpt_path = save_dir / "last_model.pth"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume not found: {resume_path}")
        print(f"[*] Resuming line training from checkpoint: {resume_path}")
        ckpt = safe_load_checkpoint(resume_path, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if args.lr is not None:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
            except Exception as e:
                print(f"    [!] Warning: Failed to load optimizer state: {e}")
        if args.amp and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print(f"    [!] Warning: Failed to load scaler state: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"    Resumed at epoch {start_epoch}, previous best Val Acc: {best_val_acc:.2f}%\n")

    num_total_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config_data = {
        "experiment_info": {
            "run_name": run_name,
            "label": getattr(args, "label", None),
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "git_commit": get_git_commit(),
            "command": " ".join(sys.argv),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "model": {
            "model_type": args.model_type,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "total_parameters": num_total_params,
            "trainable_parameters": num_trainable_params,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": lr,
            "weight_decay": args.weight_decay,
            "min_matches": args.min_matches,
            "amp": args.amp,
            "save_interval": args.save_interval,
            "save_last": args.save_last,
            "val_interval": args.val_interval,
            "num_workers": args.num_workers,
            "all_cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        "dataset": {
            "dataset_type": "LineFeatPairDataset",
            "index_path": "data/line_gt_dataset/index.json",
            "train_pairs": len(train_dataset),
            "val_pairs": len(val_dataset),
        },
        "checkpoints": {
            "run_dir": str(save_dir.resolve()),
            "best_checkpoint": None,
            "last_checkpoint": "last_model.pth" if args.save_last else None,
            "historical_checkpoints": [],
        },
        "best_validation": None,
    }
    save_config_json(config_json_path, config_data)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d} [LineTrain]",
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )

        for step, sample in enumerate(pbar):
            if args.max_steps and step >= args.max_steps:
                break
            gt_lines0 = sample["gt_lines0"][0].to(device)
            gt_lines1 = sample["gt_lines1"][0].to(device)
            M = len(gt_lines0)
            if M < 2:
                continue

            img0 = (sample["image0"].to(device) - mean) / std
            img1 = (sample["image1"].to(device) - mean) / std
            lines0 = gt_lines0.unsqueeze(0)
            lines1 = gt_lines1.unsqueeze(0)

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(img0, img1, lines0, lines1)
                loss, metrics = criterion(out["log_assignment"], num_matches=M)

            scaler.scale(loss).backward()
            if args.amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            step_loss = metrics["loss"]
            if not math.isnan(step_loss) and not math.isinf(step_loss):
                epoch_loss += step_loss
            step_count += 1

            acc_val = metrics.get('acc', 0.0)
            pbar.set_postfix_str(
                f"loss={step_loss:6.4f}, avg={epoch_loss/max(step_count, 1):6.4f}, lines={M:>2d}, acc={acc_val:5.1f}%"
            )

        epoch_sec = time.time() - t0
        train_loss = epoch_loss / max(step_count, 1)

        # Validation
        val_acc = 0.0
        val_loss = 0.0
        val_steps = 0
        is_best = False
        if epoch % args.val_interval == 0 and len(val_loader) > 0:
            model.eval()
            with torch.no_grad():
                for sample in val_loader:
                    gt_lines0 = sample["gt_lines0"][0].to(device)
                    gt_lines1 = sample["gt_lines1"][0].to(device)
                    M = len(gt_lines0)
                    if M < 2:
                        continue
                    img0 = (sample["image0"].to(device) - mean) / std
                    img1 = (sample["image1"].to(device) - mean) / std
                    lines0 = gt_lines0.unsqueeze(0)
                    lines1 = gt_lines1.unsqueeze(0)

                    with torch.amp.autocast("cuda", enabled=args.amp):
                        out = model(img0, img1, lines0, lines1)
                        loss_v, metrics_v = criterion(out["log_assignment"], num_matches=M)
                    val_loss += metrics_v["loss"]
                    val_acc += metrics_v.get("acc", 0.0)
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            avg_val_acc = val_acc / max(val_steps, 1)
            is_best = avg_val_acc >= best_val_acc
            if is_best:
                best_val_acc = avg_val_acc

            star = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) "
                  f"| Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.1f}%{star}")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | Train Loss: {train_loss:.4f}")

        # Checkpoint Management
        serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if args.amp else None,
            "best_val_acc": best_val_acc,
            "train_loss": train_loss,
            "args": serializable_args,
        }

        # 1. Update best checkpoint
        if is_best:
            torch.save(ckpt_data, best_ckpt_path)
            config_data["checkpoints"]["best_checkpoint"] = "best_model.pth"
            config_data["best_validation"] = {
                "epoch": epoch,
                "val_loss": round(avg_val_loss, 4),
                "val_acc": round(avg_val_acc, 2),
            }
            save_config_json(config_json_path, config_data)

        # 2. Update latest checkpoint
        if args.save_last:
            torch.save(ckpt_data, last_ckpt_path)

        # 3. Save periodic historical checkpoint
        if args.save_interval > 0 and (epoch % args.save_interval == 0 or epoch == args.epochs):
            epoch_ckpt_path = save_dir / f"epoch_{epoch:03d}.pth"
            torch.save(ckpt_data, epoch_ckpt_path)
            print(f"  [Checkpoint] Saved historical checkpoint -> {epoch_ckpt_path.name}")
            if epoch_ckpt_path.name not in config_data["checkpoints"]["historical_checkpoints"]:
                config_data["checkpoints"]["historical_checkpoints"].append(epoch_ckpt_path.name)
                save_config_json(config_json_path, config_data)

        # Garbage collection & CUDA cache reclamation at epoch boundaries
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config_data["experiment_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_config_json(config_json_path, config_data)
    print(f"\n[+] Saved run configuration to: {config_json_path}")


def train_homography_pipeline(args: argparse.Namespace) -> None:
    """Train FCNField on 5-channel pairs."""
    from datasets.homo_matcher_dataset import HomoMatcherDataset, collate_homo_matcher

    data_dir = Path(args.data_dir or "data/homo_matcher")
    base_save_dir = Path(args.save_dir or "checkpoints/FCN_field")
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model_type}+{data_dir.name}+【{start_time_str}】"
    save_dir = base_save_dir if args.no_timestamp else base_save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(base_save_dir, save_dir)
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else 1e-4
    train_ds = HomoMatcherDataset(data_dir, split="train", sample_points=args.line_sample_points)
    val_ds = HomoMatcherDataset(data_dir, split="val", sample_points=args.line_sample_points)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_homo_matcher, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_homo_matcher, num_workers=0)
    model = build_model(args.model_type).to(device)
    loss_fn = build_loss(args.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    config_path = save_dir / "config.json"
    config_data = {
        "experiment_info": {
            "run_name": run_name,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "git_commit": get_git_commit(),
            "command": " ".join(sys.argv),
        },
        "model": {
            "model_type": "FCN_field",
            "family": "homography",
            "feat_dim": model.feat_dim,
            "grid_size": model.grid_size,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": lr,
            "loss": "symmetry",
        },
    }
    save_config_json(config_path, config_data)
    print(f"[*] homography training  pairs={len(train_ds)} val={len(val_ds)}  device={device}")
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        train_total = len(train_loader) if args.max_steps is None else min(len(train_loader), args.max_steps)
        pbar = tqdm(
            train_loader,
            total=train_total,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]",
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )
        for batch in pbar:
            if args.max_steps is not None and steps >= args.max_steps:
                break
            moved = _batch_to_device(batch, device)
            pred = model(moved["image0"], moved["image1"])
            out = loss_fn(pred, moved)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_loss = float(out["loss"].detach())
            total += step_loss
            steps += 1
            pbar.set_postfix_str(f"symmetry={step_loss:.4f} avg={total / steps:.4f}")
        val_metrics = _homography_validate(model, loss_fn, val_loader, device, args.max_steps)
        score = val_metrics["loss"]
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] loss {total / max(steps, 1):.4f}"
            f"  val symmetry {val_metrics['symmetry']:.4f}"
        )
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, save_dir / "last_model.pth")
        if score <= best_score:
            best_score = score
            torch.save(ckpt, save_dir / "best_model.pth")
    config_data["experiment_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_config_json(config_path, config_data)
    print(f"[+] saved {save_dir}")


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _homography_validate(model, loss_fn, loader, device, max_steps) -> dict[str, float]:
    model.eval()
    totals = {"symmetry": 0.0, "loss": 0.0}
    steps = 0
    val_total = len(loader) if max_steps is None else min(len(loader), max_steps)
    pbar = tqdm(
        loader,
        total=val_total,
        desc="Val",
        bar_format=TQDM_BAR_FORMAT,
        dynamic_ncols=True,
        leave=False,
        unit="pair",
    )
    with torch.inference_mode():
        for batch in pbar:
            if max_steps is not None and steps >= max_steps:
                break
            moved = _batch_to_device(batch, device)
            pred = model(moved["image0"], moved["image1"])
            out = loss_fn(pred, moved)
            totals["symmetry"] += float(out["symmetry"])
            totals["loss"] += float(out["loss"])
            steps += 1
            pbar.set_postfix_str(f"symmetry={totals['symmetry'] / steps:.4f}")
    denom = max(steps, 1)
    return {key: value / denom for key, value in totals.items()} | {"total_pairs": steps}


def train_point_pipeline(args: argparse.Namespace) -> None:
    """Train SuperPoint with decoupled descriptor, detector, or joint stages."""
    from datasets.point_match_dataset import PointMatchDataset, collate_point_matches

    data_dir = Path(args.data_dir or "data/planar_letters")
    base_save_dir = Path(args.save_dir or "checkpoints/SuperPoint")
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d_%H%M%S")
    train_stage = getattr(args, "train_stage", "joint").lower()
    run_name = args.run_name or f"{args.model_type}_{train_stage}+{data_dir.name}+【{start_time_str}】"
    save_dir = base_save_dir if args.no_timestamp else base_save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(base_save_dir, save_dir)
    vis_dir = save_dir / "visualizations"
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else 1e-3
    train_ds = PointMatchDataset(data_dir, split="train")
    val_ds = PointMatchDataset(data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_point_matches, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_point_matches, num_workers=0)

    loss_kwargs = {
        "weight_desc": getattr(args, "weight_desc", 1.0),
        "positive_weight": getattr(args, "positive_weight", 250.0),
        "positive_margin": getattr(args, "positive_margin", 1.0),
        "negative_margin": getattr(args, "negative_margin", 0.2),
        "weight_det": getattr(args, "weight_det", 1.0),
        "det_pos_weight": getattr(args, "det_pos_weight", 50.0),
        "weight_orphan": getattr(args, "weight_orphan", 1.0),
        "weight_len": getattr(args, "weight_len", 0.5),
        "length_exponent": getattr(args, "length_exponent", 1.5),
        "weight_premature": getattr(args, "weight_premature", 0.5),
        "probe_dist": getattr(args, "probe_dist", 4.0),
        "tau_df": getattr(args, "tau_df", 0.55),
        "tau_ang": getattr(args, "tau_ang", 0.10),
        "min_length": getattr(args, "min_length", 8.0),
        "num_candidates": getattr(args, "num_candidates", 16),
        "num_line_samples": getattr(args, "num_line_samples", 10),
    }

    model = build_model(args.model_type).to(device)
    loss_fn = build_loss(args.model_type, stage=train_stage, **loss_kwargs).to(device)
    
    # 按照阶段自适应配置梯度冻结，优化器仅绑定可训练参数
    if hasattr(model, "set_train_stage"):
        model.set_train_stage(train_stage)
        trainable_params = model.get_trainable_parameters(train_stage)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)

    config_path = save_dir / "config.json"
    config_data = {
        "experiment_info": {
            "run_name": run_name,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "git_commit": get_git_commit(),
            "command": " ".join(sys.argv),
        },
        "model": {"model_type": "SuperPoint", "family": "point", "train_stage": train_stage},
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": lr,
            "loss": f"superpoint_{train_stage}",
            "trainable_param_groups": len(trainable_params),
            "loss_weights": loss_kwargs,
        },
    }
    save_config_json(config_path, config_data)
    print(f"[*] SuperPoint stage: '{train_stage}' | pairs={len(train_ds)} val={len(val_ds)} | trainable params: {len(trainable_params)} tensors | device={device}")
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        train_total = len(train_loader) if args.max_steps is None else min(len(train_loader), args.max_steps)
        pbar = tqdm(
            train_loader,
            total=train_total,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d} [{train_stage.capitalize()}]",
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )
        for batch in pbar:
            if args.max_steps is not None and steps >= args.max_steps:
                break
            moved = _batch_to_device(batch, device)
            pred0 = model(moved["image0"])
            pred1 = model(moved["image1"])
            out = loss_fn(
                pred0,
                pred1,
                moved["homography"],
                image0=moved["image0"],
                image1=moved["image1"],
                points0=moved.get("points0"),
                points1=moved.get("points1"),
            )
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            step_loss = float(out["loss"].detach())
            total += step_loss
            steps += 1
            if train_stage == "descriptor":
                pbar.set_postfix_str(f"loss={step_loss:.4f} ld={float(out['ld']):.3f}")
            elif train_stage == "detector":
                pbar.set_postfix_str(
                    f"loss={step_loss:.4f} det={float(out['l_det']):.3f} orph={float(out['l_orphan']):.3f} len={float(out['l_len']):.3f}"
                )
            else:
                pbar.set_postfix_str(
                    f"loss={step_loss:.4f} desc={float(out.get('ld', 0.0)):.3f} orph={float(out.get('l_orphan', 0.0)):.3f} len={float(out.get('l_len', 0.0)):.3f}"
                )
        val_metrics = _point_validate(model, loss_fn, val_loader, device, args.max_steps, vis_dir, epoch, train_stage=train_stage)
        score = val_metrics["loss"]
        if train_stage == "detector":
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] loss {total / max(steps, 1):.4f}"
                f"  val det {val_metrics['loss']:.4f}  lines {val_metrics.get('num_lines', 0.0):.1f}  orphan {val_metrics.get('orphan_rate', 0.0)*100:.1f}%"
            )
        elif train_stage == "descriptor":
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] loss {total / max(steps, 1):.4f}"
                f"  val desc {val_metrics['loss']:.4f}  match {val_metrics.get('match_acc', 0.0) * 100:.1f}%"
            )
        else:
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] loss {total / max(steps, 1):.4f}"
                f"  val loss {val_metrics['loss']:.4f}  match {val_metrics.get('match_acc', 0.0) * 100:.1f}%  lines {val_metrics.get('num_lines', 0.0):.1f}"
            )
        ckpt = {
            "epoch": epoch,
            "train_stage": train_stage,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, save_dir / "last_model.pth")
        if score <= best_score:
            best_score = score
            torch.save(ckpt, save_dir / "best_model.pth")
    config_data["experiment_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_config_json(config_path, config_data)
    print(f"[+] saved {save_dir}")


def _point_validate(
    model,
    loss_fn,
    loader,
    device,
    max_steps,
    vis_dir: Path,
    epoch: int,
    train_stage: str = "joint",
) -> dict[str, float]:
    from visualization.point_matches import (
        match_accuracy,
        extract_detected_lines,
        save_detection_mask_lines_figure,
        save_descriptor_matching_figure,
    )

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_lines = 0.0
    total_orphan_rate = 0.0
    steps = 0
    val_total = len(loader) if max_steps is None else min(len(loader), max_steps)
    pbar = tqdm(
        loader,
        total=val_total,
        desc=f"Val [{train_stage.capitalize()}]",
        bar_format=TQDM_BAR_FORMAT,
        dynamic_ncols=True,
        leave=False,
        unit="pair",
    )
    with torch.inference_mode():
        for batch in pbar:
            if max_steps is not None and steps >= max_steps:
                break
            moved = _batch_to_device(batch, device)
            pred0 = model(moved["image0"])
            pred1 = model(moved["image1"])
            out = loss_fn(
                pred0,
                pred1,
                moved["homography"],
                image0=moved["image0"],
                image1=moved["image1"],
                points0=moved.get("points0"),
                points1=moved.get("points1"),
            )
            step_loss = float(out["loss"])
            total_loss += step_loss

            accuracy = 0.0
            if "descriptor" in pred0 and moved.get("points0") is not None and moved.get("points1") is not None:
                accuracy, _ = match_accuracy(pred0["descriptor"], pred1["descriptor"], moved["points0"], moved["points1"])
                total_acc += accuracy

            pair_lines = 0.0
            pair_orphan = 0.0
            if train_stage in ("detector", "joint") and "heatmap" in pred0 and moved.get("image0") is not None:
                geom0 = extract_detected_lines(pred0["heatmap"], moved["image0"][:, 3:4], moved["image0"][:, 4:5], loss_fn)
                pair_lines = float(geom0["num_lines"])
                pair_orphan = float(geom0["orphan_rate"])
                total_lines += pair_lines
                total_orphan_rate += pair_orphan

            steps += 1
            if train_stage == "detector":
                pbar.set_postfix_str(f"det={total_loss / steps:.4f} lines={total_lines / steps:.1f}")
            elif train_stage == "descriptor":
                pbar.set_postfix_str(f"desc={total_loss / steps:.4f} match={100 * total_acc / steps:.1f}%")
            else:
                pbar.set_postfix_str(f"loss={total_loss / steps:.4f} match={100 * total_acc / steps:.1f}% lines={total_lines / steps:.1f}")

            # 验证可视化 (前 4 对样本分别保存对应图像)
            if steps <= 4:
                pair_id = batch["pair_id"]
                # 1. 检测验证: 原始掩膜 + 直线 (*_detection.jpg)
                if train_stage in ("detector", "joint"):
                    save_detection_mask_lines_figure(
                        vis_dir / f"epoch_{epoch:02d}_{pair_id}_detection.jpg",
                        batch["bgr0"].numpy(),
                        batch["bgr1"].numpy(),
                        pred0,
                        pred1,
                        moved["image0"][:, 3:4],
                        moved["image1"][:, 3:4],
                        moved["image0"][:, 4:5],
                        moved["image1"][:, 4:5],
                        loss_fn,
                        pair_id=pair_id,
                        metrics_summary={"loss": step_loss},
                        points0=moved.get("points0"),
                        points1=moved.get("points1"),
                    )

                # 2. 描述验证: 配对情况 (*_matching.jpg)
                if train_stage in ("descriptor", "joint") and moved.get("points0") is not None and moved.get("points1") is not None:
                    save_descriptor_matching_figure(
                        vis_dir / f"epoch_{epoch:02d}_{pair_id}_matching.jpg",
                        batch["bgr0"].numpy(),
                        batch["bgr1"].numpy(),
                        pred0["descriptor"],
                        pred1["descriptor"],
                        moved["points0"],
                        moved["points1"],
                        moved["homography"],
                        pair_id=pair_id,
                        metrics_summary={"ld": float(out.get("ld", step_loss))},
                    )

    denom = max(steps, 1)
    return {
        "loss": total_loss / denom,
        "match_acc": total_acc / denom,
        "num_lines": total_lines / denom,
        "orphan_rate": total_orphan_rate / denom,
        "total_pairs": steps,
    }


def main() -> None:
    from factory_args import warn_unused_arguments

    args, parser = parse_args()
    warn_unused_arguments(parser, train_applicable_arguments(args.model_type), args.model_type)
    family = model_family(args.model_type)
    if family == "object":
        train_object_pipeline(args)
    elif family == "homography":
        train_homography_pipeline(args)
    elif family == "point":
        train_point_pipeline(args)
    else:
        train_line_pipeline(args)


if __name__ == "__main__":
    main()
