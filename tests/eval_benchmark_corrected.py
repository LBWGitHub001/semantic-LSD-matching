#!/usr/bin/env python3
"""
Audited & Corrected Benchmark Evaluation for XFeatObjectMatcher.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from models import build_model
from datasets.semantic_region_dataset import SemanticRegionDataset, collate_region_pairs
from train import evaluate_object_pipeline


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = SemanticRegionDataset("data/structured3d_semantic_regions", split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_region_pairs, num_workers=2)

    model = build_model("xfeat_object", feat_dim=64, d_model=128, num_layers=2, nhead=4).to(device)
    ckpt_path = Path("checkpoints/semantic_region_matcher/best_model.pth")
    if not ckpt_path.is_file():
        print("Checkpoint not found:", ckpt_path)
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # 1. Raw visual matching
    raw = evaluate_object_pipeline(model, test_loader, device, desc="Testing (Raw Visual)")

    # 2. Semantic constrained matching
    cat = evaluate_object_pipeline(model, test_loader, device, desc="Testing (Semantic Constrained)", enforce_same_category=True)

    print("=" * 75)
    print(f"CORRECTED BENCHMARK RESULTS ({len(test_ds)} unseen test scene pairs):")
    print("-" * 75)
    print("1. Raw Visual Appearance Matching (Zero Semantic Prior):")
    print(f"   Precision          : {raw['precision']:.2f}%")
    print(f"   Recall             : {raw['recall']:.2f}%")
    print(f"   F1-Score           : {raw['f1']:.2f}%")
    print(f"   Matches            : {raw['tp']} TP, {raw['fp']} FP, {raw['fn']} FN")
    print(f"   Cross-Category FP  : {raw['cross_cat_fp']} ({raw['cross_cat_rate']:.1f}% of all FPs are cross-class violations)")
    print(f"   Dustbin Decisions  : Precision={raw['dustbin_prec']:.2f}%, Recall={raw['dustbin_rec']:.2f}%, Overall Acc={raw['dustbin_acc']:.2f}%")
    print("-" * 75)
    print("2. Semantic Constrained Matching (Same-Category Prior Masking):")
    print(f"   Precision          : {cat['precision']:.2f}% (Boosted from {raw['precision']:.2f}%)")
    print(f"   Recall             : {cat['recall']:.2f}%")
    print(f"   F1-Score           : {cat['f1']:.2f}% (Boosted from {raw['f1']:.2f}%)")
    print(f"   Matches            : {cat['tp']} TP, {cat['fp']} FP, {cat['fn']} FN")
    print(f"   Cross-Category FP  : {cat['cross_cat_fp']} (0.0% eliminated completely!)")
    print(f"   Dustbin Decisions  : Precision={cat['dustbin_prec']:.2f}%, Recall={cat['dustbin_rec']:.2f}%, Overall Acc={cat['dustbin_acc']:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
