#!/usr/bin/env python3
"""
Interactive Semantic Region Patch Matcher Verification Tool.

Verifies model predictions on specific scene pairs from the Structured3D dataset:
- Prints ground-truth vs prediction matching tables with probabilities.
- Reports TP, FP, FN, Precision, Recall, and Dustbin decisions.
- Exports rich visualization showing:
  1. Full-image semantic mask overlays (Alpha Blending + Contours).
  2. The actual 112x112 small region image patches that entered the network.
  3. Green solid lines for correct matches, red solid lines for false alarms, yellow dashed for missed items.

Usage:
  python tools/verify_semantic_matches.py --sample-idx 0
  python tools/verify_semantic_matches.py --pair-id scene_00091_968148_v0_v3
  python tools/verify_semantic_matches.py --all-test-summary
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from models import build_model
from datasets.semantic_region_dataset import (
    SemanticRegionDataset,
    collate_region_pairs,
    visualize_semantic_region_matches,
)
from datasets.object_dataset import predict_bipartite_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Semantic Region Patch Matching Model")
    parser.add_argument("--data-dir", type=str, default="data/structured3d_semantic_regions", help="Dataset directory")
    parser.add_argument("--ckpt", type=str, default="checkpoints/semantic_patch_matcher/best_model.pth", help="Checkpoint path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample pair to inspect (default: 0)")
    parser.add_argument("--pair-id", type=str, default=None, help="Specific pair_id to inspect")
    parser.add_argument("--thresh", type=float, default=0.2, help="Bipartite match probability threshold")
    parser.add_argument("--enforce-category", action="store_true", default=False, help="Enforce same-category matching prior")
    parser.add_argument("--out-dir", type=str, default="outputs/verification", help="Output directory for visual verification cards")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def render_patch_strip(
    patches: torch.Tensor,
    labels: list[str],
    matched_indices: set[int],
    title: str = "View Patches",
) -> np.ndarray:
    """
    Renders a vertical or horizontal strip of the 112x112 small region patches.
    patches: (M, 3, 112, 112) float32 normalized ImageNet tensor
    """
    M = patches.shape[0]
    if M == 0:
        blank = np.zeros((140, 140, 3), dtype=np.uint8)
        cv2.putText(blank, "No Patches", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return blank

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    patch_cards = []
    p_np = patches.detach().cpu().numpy()

    for i in range(M):
        p_chw = p_np[i]
        p_hwc = p_chw.transpose(1, 2, 0)
        # Denormalize
        bgr = np.clip((p_hwc * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255)[..., ::-1].astype(np.uint8)

        card = np.full((140, 116, 3), 40, dtype=np.uint8)
        card[20:132, 2:114] = bgr

        # Border color: Green if matched, Gray if unmatched
        is_m = i in matched_indices
        border_col = (40, 210, 40) if is_m else (100, 100, 100)
        cv2.rectangle(card, (2, 20), (113, 131), border_col, 2)

        lbl = labels[i] if i < len(labels) else f"#{i}"
        # Truncate label if too long
        lbl_short = lbl[:10]
        cv2.putText(card, f"#{i}:{lbl_short}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
        patch_cards.append(card)

    strip = np.hstack(patch_cards)
    return strip


def verify_pair(
    model: torch.nn.Module,
    dataset: SemanticRegionDataset,
    sample_idx: int,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    item = dataset[sample_idx]
    batch = collate_region_pairs([item])

    patches0 = batch["patches0"].to(device)
    patches1 = batch["patches1"].to(device)

    with torch.no_grad():
        out = model(patches0, patches1)
        prob_matrix = out["prob_matrix"][0]
        P_np = prob_matrix.detach().cpu().numpy()

    M0 = patches0.shape[1]
    M1 = patches1.shape[1]

    cats0 = [o.get("category", "") for o in batch["objs0"]]
    cats1 = [o.get("category", "") for o in batch["objs1"]]
    labels0 = [o.get("label", str(i)) for i, o in enumerate(batch["objs0"])]
    labels1 = [o.get("label", str(j)) for j, o in enumerate(batch["objs1"])]

    filter_cats0 = cats0 if args.enforce_category else None
    filter_cats1 = cats1 if args.enforce_category else None

    pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(
        prob_matrix,
        thresh=args.thresh,
        categories_0=filter_cats0,
        categories_1=filter_cats1,
    )

    gt_matches = batch["gt_matches"][0]
    gt_set = set(gt_matches)
    pred_set = set(pred_matches)

    tp_set = pred_set & gt_set
    fp_set = pred_set - gt_set
    fn_set = gt_set - pred_set

    gt_un0 = set(batch["unmatched_0"][0])
    gt_un1 = set(batch["unmatched_1"][0])

    pair_id = batch["pair_id"]
    rot = batch.get("rotation_deg", 0.0)
    base = batch.get("baseline_m", 0.0)

    print("\n" + "=" * 78)
    print(f"VERIFICATION REPORT: {pair_id} (Split: {args.split}, Sample #{sample_idx})")
    print(f"  • View Rotation: {rot:.1f}° | Camera Baseline: {base:.2f} m")
    print(f"  • Input: View0 has {M0} region patches, View1 has {M1} region patches (112x112 each)")
    print(f"  • Ground Truth Matches: {len(gt_matches)} pairs | Out-of-view Dustbin: View0={len(gt_un0)}, View1={len(gt_un1)}")
    print(f"  • Matching Mode: {'Semantic Class Constrained' if args.enforce_category else 'Raw Visual Appearance (Unconstrained)'}")
    print("-" * 78)

    # 1. Ground Truth Audit Table
    print("[1] GROUND TRUTH CORRESPONDENCES & MODEL CONFIDENCE:")
    print(f"  {'Pair':<12} | {'View 0 Region':<18} | {'View 1 Region':<18} | {'Model Prob':<11} | {'Status'}")
    print("  " + "-" * 74)
    for i, j in gt_matches:
        l0 = labels0[i] if i < len(labels0) else f"#{i}"
        l1 = labels1[j] if j < len(labels1) else f"#{j}"
        prob = P_np[i, j] if i < M0 and j < M1 else 0.0
        status = "✓ TRUE POSITIVE" if (i, j) in pred_set else "✗ MISSED (FN)"
        pair_str = f"({i} <-> {j})"
        print(f"  {pair_str:<12} | {l0:<18} | {l1:<18} | {prob:.4f}     | {status}")

    # 2. Predictions Table
    print("\n[2] MODEL PREDICTIONS AUDIT:")
    if not pred_matches:
        print("  (No matches predicted above threshold)")
    else:
        print(f"  {'Pair':<12} | {'View 0 Region':<18} | {'View 1 Region':<18} | {'Model Prob':<11} | {'Evaluation'}")
        print("  " + "-" * 74)
        for i, j in pred_matches:
            l0 = labels0[i] if i < len(labels0) else f"#{i}"
            l1 = labels1[j] if j < len(labels1) else f"#{j}"
            prob = P_np[i, j] if i < M0 and j < M1 else 0.0
            if (i, j) in gt_set:
                tag = "✓ CORRECT (TP)"
            else:
                c0 = cats0[i] if i < len(cats0) else ""
                c1 = cats1[j] if j < len(cats1) else ""
                cross_tag = " [CLASS MISMATCH!]" if c0 != c1 else " [Same Class Confusion]"
                tag = f"✗ FALSE POSITIVE{cross_tag}"
            pair_str = f"({i} <-> {j})"
            print(f"  {pair_str:<12} | {l0:<18} | {l1:<18} | {prob:.4f}     | {tag}")

    # 3. Summary Performance Metrics for this pair
    prec = len(tp_set) / (len(tp_set) + len(fp_set) + 1e-6) * 100.0 if pred_set else 0.0
    rec = len(tp_set) / (len(gt_set) + 1e-6) * 100.0 if gt_set else 100.0
    f1 = 2.0 * prec * rec / (prec + rec + 1e-6) if (prec + rec) > 0 else 0.0

    print("-" * 78)
    print(f"PAIR SCORE SUMMARY:")
    print(f"  • Correct Matches (TP): {len(tp_set)}")
    print(f"  • Wrong Matches   (FP): {len(fp_set)}")
    print(f"  • Missed Matches  (FN): {len(fn_set)}")
    print(f"  • Precision: {prec:5.1f}% | Recall: {rec:5.1f}% | F1: {f1:5.1f}%")
    print("=" * 78)

    # 4. Generate & Save Visual Verification Cards
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Card 1: Full-Image Mask Overlay with Matching Lines
    mode_tag = "cat_constrained" if args.enforce_category else "raw_visual"
    vis_file = out_dir / f"verify_pair_{sample_idx:03d}_{pair_id}_{mode_tag}.jpg"
    visualize_semantic_region_matches(
        bgr0=batch["raw_img0"],
        bgr1=batch["raw_img1"],
        masks0=batch["masks0"],
        masks1=batch["masks1"],
        regs0=batch.get("regs0", []),
        regs1=batch.get("regs1", []),
        pred_matches=pred_matches,
        gt_matches=gt_matches,
        pred_un0=pred_un0,
        pred_un1=pred_un1,
        out_path=vis_file,
        title=f"Sample #{sample_idx}: {pair_id} (Rot={rot:.1f}deg, Base={base:.2f}m)",
    )
    print(f"[✓] Full-Image Mask Verification Card saved -> {vis_file}")

    # Card 2: Patch Collage Card (Visualizing the actual 112x112 patches that entered the network)
    patch_vis_file = out_dir / f"patches_{sample_idx:03d}_{pair_id}.jpg"
    matched_i_set = {i for i, _ in pred_matches}
    matched_j_set = {j for _, j in pred_matches}

    strip0 = render_patch_strip(patches0[0], labels0, matched_i_set, title="View 0 Patches")
    strip1 = render_patch_strip(patches1[0], labels1, matched_j_set, title="View 1 Patches")

    max_w = max(strip0.shape[1], strip1.shape[1])
    collage = np.zeros((360, max_w + 40, 3), dtype=np.uint8)
    collage.fill(30)

    cv2.putText(collage, f"View 0 Region Patches (112x112, {M0} patches):", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    collage[30 : 30 + strip0.shape[0], 10 : 10 + strip0.shape[1]] = strip0

    cv2.putText(collage, f"View 1 Region Patches (112x112, {M1} patches):", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    collage[205 : 205 + strip1.shape[0], 10 : 10 + strip1.shape[1]] = strip1

    cv2.imwrite(str(patch_vis_file), collage)
    print(f"[✓] Patch Card Strip saved -> {patch_vis_file}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=================================================================")
    print("      Semantic Region Patch Matcher: Interactive Verifier        ")
    print("=================================================================")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Dataset:    {args.data_dir} (Split: {args.split})")
    print(f"  Device:     {device}")
    print("=================================================================\n")

    dataset = SemanticRegionDataset(args.data_dir, split=args.split)
    model = build_model("xfeat_object", feat_dim=64, d_model=128, num_layers=2, nhead=4).to(device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    sample_idx = args.sample_idx
    if args.pair_id:
        found = False
        for idx in range(len(dataset)):
            if dataset.pair_ids[idx] == args.pair_id:
                sample_idx = idx
                found = True
                break
        if not found:
            print(f"Warning: pair_id '{args.pair_id}' not found. Defaulting to sample_idx 0.")

    verify_pair(model, dataset, sample_idx, args)


if __name__ == "__main__":
    main()
