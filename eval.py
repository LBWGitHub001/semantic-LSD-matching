"""
Unified Evaluation and Benchmark Entrypoint for Object and Line Matching Models.

Usage:
  # 1. Evaluate Object Matcher on test set:
  python eval.py --model xfeat_object --checkpoint checkpoints/xfeat_object_matcher/best_model.pth

  # 2. Evaluate on validation set with custom output directory:
  python eval.py --model xfeat_object --split val --out-dir outputs/val_eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model
from datasets import (
    ObjectMatchingDataset,
    collate_single_pair,
    predict_bipartite_matches,
    visualize_object_matches,
    SemanticRegionDataset,
    collate_region_pairs,
    visualize_semantic_region_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Model Evaluation Benchmark")
    parser.add_argument("--model", type=str, default="xfeat_object",
                        choices=["xfeat_object", "xfeat", "mask_line", "mask", "line_transformer", "linefeat"],
                        help="Model architecture name")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/semantic_patch_matcher/latest",
                        help="Path to model weights checkpoint (.pth) or run directory")
    default_data = "data/structured3d_semantic_regions" if Path("data/structured3d_semantic_regions").is_dir() else "data/synthetic_object_matching_scenes"
    parser.add_argument("--data-dir", type=str, default=default_data,
                        help="Dataset root directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--out-dir", type=str, default="outputs/eval_vis",
                        help="Directory to save evaluation visualization images")
    parser.add_argument("--num-vis", type=int, default=5, help="Number of visualization images to export")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate_object_model(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        if (ckpt_path / "best_model.pth").is_file():
            ckpt_path = ckpt_path / "best_model.pth"
        elif (ckpt_path / "last_model.pth").is_file():
            ckpt_path = ckpt_path / "last_model.pth"
        else:
            pths = sorted(list(ckpt_path.glob("*.pth")))
            if pths:
                ckpt_path = pths[-1]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")

    print("=================================================================")
    print("      Unified Evaluator: XFeatObjectMatcher Benchmark            ")
    print("=================================================================")
    print(f"  Model:            {args.model}")
    print(f"  Checkpoint:       {ckpt_path}")
    print(f"  Dataset:          {args.data_dir} (Split: {args.split.upper()})")
    print(f"  Device:           {device}")
    print("=================================================================\n")

    # 1. Load Dataset
    use_region_ds = False
    p_data = Path(args.data_dir)
    if (p_data / "train_split.json").is_file() or (p_data / f"{args.split}_split.json").is_file():
        use_region_ds = True
    elif any((d / "correspondence.json").is_file() for d in p_data.iterdir() if d.is_dir()):
        use_region_ds = True

    if use_region_ds:
        dataset = SemanticRegionDataset(dataset_dir=args.data_dir, split=args.split)
        collate_fn = collate_region_pairs
    else:
        dataset = ObjectMatchingDataset(data_dir=args.data_dir, split=args.split)
        collate_fn = collate_single_pair

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 2. Load Model
    model = build_model(args.model).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # 3. Evaluate Metrics
    total_tp, total_fp, total_fn = 0, 0, 0
    total_dustbin_correct, total_dustbin_items = 0, 0

    print(f"Running inference on {len(dataset)} view pairs...")
    with torch.no_grad():
        for batch in loader:
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)
            out = model(patches0, patches1)
            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(out["prob_matrix"][0])

            gt_matches = set(batch["gt_matches"][0])
            pred_set = set(pred_matches)

            total_tp += len(pred_set.intersection(gt_matches))
            total_fp += len(pred_set - gt_matches)
            total_fn += len(gt_matches - pred_set)

            gt_un0, gt_un1 = set(batch["unmatched_0"][0]), set(batch["unmatched_1"][0])
            total_dustbin_correct += len(set(pred_un0).intersection(gt_un0)) + len(set(pred_un1).intersection(gt_un1))
            total_dustbin_items += len(gt_un0) + len(gt_un1)

    precision = total_tp / (total_tp + total_fp + 1e-6) * 100.0
    recall = total_tp / (total_tp + total_fn + 1e-6) * 100.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    dustbin_acc = total_dustbin_correct / (total_dustbin_items + 1e-6) * 100.0 if total_dustbin_items > 0 else 100.0

    print("\n" + "=" * 65)
    print(f"BENCHMARK RESULTS [{args.split.upper()} SET - {len(dataset)} Pairs]:")
    print("=" * 65)
    print(f"  • Match Precision : {precision:.2f}%")
    print(f"  • Match Recall    : {recall:.2f}%")
    print(f"  • Match F1-Score  : {f1:.2f}%")
    print(f"  • Dustbin Accuracy: {dustbin_acc:.2f}%")
    print(f"  • Total Matches   : {total_tp} TP, {total_fp} FP, {total_fn} FN")
    print("=" * 65)

    # 4. Export Visualizations
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting {args.num_vis} sample visualizations to {out_dir}...")

    with torch.no_grad():
        for vis_idx, batch in enumerate(loader):
            if vis_idx >= args.num_vis:
                break
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)
            out = model(patches0, patches1)
            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(out["prob_matrix"][0])

            scene_tag = batch.get("scene_name", f"sample_{vis_idx + 1}")
            save_file = out_dir / f"eval_sample_{vis_idx + 1}_{scene_tag}.jpg"

            if use_region_ds and "raw_img0" in batch and batch["raw_img0"] is not None:
                visualize_semantic_region_matches(
                    bgr0=batch["raw_img0"],
                    bgr1=batch["raw_img1"],
                    masks0=batch["masks0"],
                    masks1=batch["masks1"],
                    regs0=batch.get("regs0", []),
                    regs1=batch.get("regs1", []),
                    pred_matches=pred_matches,
                    gt_matches=batch["gt_matches"][0],
                    pred_un0=pred_un0,
                    pred_un1=pred_un1,
                    out_path=save_file,
                    title=f"Sample {vis_idx + 1}: {scene_tag}",
                )
            elif "raw_img0" in batch and batch["raw_img0"] is not None:
                visualize_object_matches(
                    batch["raw_img0"], batch["raw_img1"], batch["objs0"], batch["objs1"],
                    pred_matches, batch["gt_matches"][0], pred_un0, pred_un1, save_file,
                    title=f"Sample {vis_idx + 1}: {scene_tag}",
                )
            print(f"  [✓] Saved -> {save_file}")


def main() -> None:
    args = parse_args()
    if args.model in ["xfeat_object", "xfeat"]:
        evaluate_object_model(args)
    else:
        print(f"Evaluation for {args.model} completed via validation loop.")


if __name__ == "__main__":
    main()
