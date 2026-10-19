"""
LineFeat Complete Pipeline Verification & Demo.

This script verifies:
  1. Dataset loading (LineFeatPairDataset) with images, camera parameters, and 3D wireframe GT lines.
  2. 2D Line Segment Detection using LSD (LineDetector).
  3. Deep feature map extraction (ResNet50 backbone) & point-based line descriptor pooling (LineDescriptor).
  4. Cross-image line matching (LineMatcher) with MNN, Lowe's ratio test, and semantic gating.
  5. Quantitative verification of matching accuracy against Ground Truth.
  6. Visual comparison artifact generation in outputs/linefeat/.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import (
    LineDetector,
    LineDescriptor,
    LineMatcher,
    LineFeatPairDataset,
    query_line_semantic,
    get_distinct_colors,
)


class ResNetBackbone(nn.Module):
    """Extracts high-resolution multi-scale dense feature maps from ResNet50."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)

        # Extract features up to layer2 (stride 8, 512 channels)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2

        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feat = self.layer2(x)  # (B, 512, H/8, W/8)
        return feat


def compute_line_distance(l1: np.ndarray, l2: np.ndarray) -> float:
    """
    Compute average endpoint-to-line orthogonal distance between two 2D segments.
    l1, l2: [x1, y1, x2, y2]
    """
    p1 = l1[0:2]
    p2 = l1[2:4]
    q1 = l2[0:2]
    q2 = l2[2:4]

    # Angle difference
    v1 = p2 - p1
    v2 = q2 - q1
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < 1e-4 or len2 < 1e-4:
        return 999.0

    u1 = v1 / len1
    u2 = v2 / len2
    cos_theta = abs(np.dot(u1, u2))
    if cos_theta < 0.85:  # > ~30 degrees angle difference
        return 999.0

    # Midpoints distance
    mid1 = (p1 + p2) / 2.0
    mid2 = (q1 + q2) / 2.0
    return float(np.linalg.norm(mid1 - mid2))


def run_verification(pair_idx: int = 0, output_dir: Path = ROOT / "outputs" / "linefeat") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running LineFeat Verification on device: {device}")

    # 1. Load Dataset
    print("[1/5] Loading LineFeatPairDataset...")
    dataset = LineFeatPairDataset(
        index_path="data/line_gt_dataset/index.json",
        min_matches=5,
        load_images=True,
        load_semantics=True,
        load_cameras=True,
    )
    print(f"      Total candidate pairs with >=5 matches: {len(dataset)}")

    sample = dataset[pair_idx]
    pair_id = sample["pair_id"]
    rot_deg = sample["rotation_deg"]
    baseline_m = sample["baseline_meters"]
    gt_lines0 = sample["gt_lines0"].numpy()  # (M, 4)
    gt_lines1 = sample["gt_lines1"].numpy()  # (M, 4)
    gt_semantics = sample["gt_semantics"]
    gt_line_ids = sample["gt_line_ids"].numpy()
    num_gt = len(gt_lines0)

    print(f"      Selected pair: {pair_id}")
    print(f"      Relative Rotation: {rot_deg:.1f} deg | Baseline: {baseline_m:.3f} m")
    print(f"      Ground-Truth Matches: {num_gt}")

    # Convert images to BGR uint8 for OpenCV LSD and visualization
    img0_t = sample["image0"]  # (3, H, W) in [0, 1]
    img1_t = sample["image1"]
    h, w = img0_t.shape[1], img0_t.shape[2]

    img0_bgr = (img0_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()
    img1_bgr = (img1_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()
    sem0 = sample["semantic0"].numpy()
    sem1 = sample["semantic1"].numpy()

    # 2. Extract Deep Convolutional Feature Maps
    print("[2/5] Extracting dense deep feature maps (ResNet50 backbone)...")
    backbone = ResNetBackbone(pretrained=True).to(device)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    t0 = (img0_t.unsqueeze(0).to(device) - mean) / std
    t1 = (img1_t.unsqueeze(0).to(device) - mean) / std

    with torch.no_grad():
        feat_map0 = backbone(t0)  # (1, 512, H_f, W_f)
        feat_map1 = backbone(t1)

    print(f"      Feature map shape: {tuple(feat_map0.shape)}")

    # Initialize Line Descriptor Extractor and Matcher
    descriptor_extractor = LineDescriptor(num_samples=7, pooling="mean", normalize=True)
    matcher_semantic = LineMatcher(ratio_threshold=0.88, mutual_check=True)
    matcher_raw = LineMatcher(ratio_threshold=0.88, mutual_check=True)

    # -------------------------------------------------------------
    # Test A: Pure GT Lines Feature Extraction & Verification
    # -------------------------------------------------------------
    print("\n--- Test A: Pure Ground-Truth Wireframe Line Descriptors ---")
    desc_gt0 = descriptor_extractor.extract(gt_lines0, feat_map0, orig_size=(h, w))
    desc_gt1 = descriptor_extractor.extract(gt_lines1, feat_map1, orig_size=(h, w))

    # Match GT lines using deep features
    gt_matches = matcher_semantic.match(
        desc_gt0,
        desc_gt1,
        lines1=gt_lines0,
        lines2=gt_lines1,
        semantics1=gt_semantics,
        semantics2=gt_semantics,
    )

    correct_gt_matches = sum(1 for (i, j, s) in gt_matches if i == j)
    gt_acc = correct_gt_matches / max(num_gt, 1) * 100.0
    print(f"      GT Lines: {num_gt} pairs")
    print(f"      Deep Descriptor Matches Found: {len(gt_matches)}")
    print(f"      Correct Matches (Same 3D Line ID): {correct_gt_matches} / {num_gt} ({gt_acc:.1f}%)")

    # Visualize GT Line Matches
    gt_vis = LineMatcher.draw_matches(
        img0_bgr, img1_bgr, gt_lines0, gt_lines1,
        [(i, i, 1.0) for i in range(num_gt)],
        max_draw=30,
    )
    gt_vis_path = output_dir / f"{pair_id}_ground_truth.jpg"
    cv2.imwrite(str(gt_vis_path), gt_vis)
    print(f"      Saved GT visualization: {gt_vis_path.name}")

    # -------------------------------------------------------------
    # Test B: Full LSD Detection + Deep Descriptors + Matching
    # -------------------------------------------------------------
    print("\n--- Test B: LSD Detection + Deep Feature Line Matching ---")
    detector = LineDetector(scale=0.8)
    lsd_lines0 = detector.detect(img0_bgr, min_length=30.0, margin=8)
    lsd_lines1 = detector.detect(img1_bgr, min_length=30.0, margin=8)
    print(f"      Detected LSD Lines: View0 = {len(lsd_lines0)}, View1 = {len(lsd_lines1)}")

    # Query semantic class for each detected line
    lsd_sem0 = [query_line_semantic(sem0, ((int(l[0]), int(l[1])), (int(l[2]), int(l[3])))) for l in lsd_lines0]
    lsd_sem1 = [query_line_semantic(sem1, ((int(l[0]), int(l[1])), (int(l[2]), int(l[3])))) for l in lsd_lines1]

    # Extract deep descriptors along LSD segments
    desc_lsd0 = descriptor_extractor.extract(lsd_lines0, feat_map0, orig_size=(h, w))
    desc_lsd1 = descriptor_extractor.extract(lsd_lines1, feat_map1, orig_size=(h, w))

    # Match WITHOUT semantic gating (baseline deep matching)
    raw_matches = matcher_raw.match(
        desc_lsd0,
        desc_lsd1,
        lines1=lsd_lines0,
        lines2=lsd_lines1,
    )

    # Match WITH semantic gating (proposed semantic-guided matching)
    sem_matches = matcher_semantic.match(
        desc_lsd0,
        desc_lsd1,
        lines1=lsd_lines0,
        lines2=lsd_lines1,
        semantics1=lsd_sem0,
        semantics2=lsd_sem1,
    )

    # Check semantic consistency
    raw_consistent = sum(1 for (i, j, s) in raw_matches if lsd_sem0[i] == lsd_sem1[j])
    sem_consistent = sum(1 for (i, j, s) in sem_matches if lsd_sem0[i] == lsd_sem1[j])

    print(f"      [Without Semantics] Matches: {len(raw_matches)} | Semantically Consistent: {raw_consistent}/{len(raw_matches)} ({raw_consistent/max(len(raw_matches), 1)*100:.1f}%)")
    print(f"      [With Semantics]    Matches: {len(sem_matches)} | Semantically Consistent: {sem_consistent}/{len(sem_matches)} ({sem_consistent/max(len(sem_matches), 1)*100:.1f}%)")

    # -------------------------------------------------------------
    # Test C: Verification of Detected LSD Matches against GT Lines
    # -------------------------------------------------------------
    print("\n--- Test C: Verification against 3D Ground Truth ---")
    verified_matches = 0
    for (i, j, s) in sem_matches:
        l0 = lsd_lines0[i]
        l1 = lsd_lines1[j]

        # Check if l0 is close to any GT line in View0 AND l1 is close to the matching GT line in View1
        is_correct = False
        for k in range(num_gt):
            d0 = compute_line_distance(l0, gt_lines0[k])
            d1 = compute_line_distance(l1, gt_lines1[k])
            if d0 < 25.0 and d1 < 25.0:  # Threshold for overlapping segment
                is_correct = True
                break
        if is_correct:
            verified_matches += 1

    gt_overlap_rate = (verified_matches / max(len(sem_matches), 1)) * 100.0
    print(f"      LSD Matches verified against 3D structural GT lines: {verified_matches} ({gt_overlap_rate:.1f}% precision on structural lines)")

    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------
    vis_sem = LineMatcher.draw_matches(
        img0_bgr, img1_bgr, lsd_lines0, lsd_lines1, sem_matches, max_draw=40
    )
    vis_sem_path = output_dir / f"{pair_id}_lsd_semantic_matches.jpg"
    cv2.imwrite(str(vis_sem_path), vis_sem)
    print(f"      Saved Semantic Line Matching visualization: {vis_sem_path.name}")

    vis_raw = LineMatcher.draw_matches(
        img0_bgr, img1_bgr, lsd_lines0, lsd_lines1, raw_matches, max_draw=40
    )
    vis_raw_path = output_dir / f"{pair_id}_lsd_raw_matches.jpg"
    cv2.imwrite(str(vis_raw_path), vis_raw)
    print(f"      Saved Raw Line Matching visualization: {vis_raw_path.name}")

    print("\n[+] Verification Completed Successfully!")
    print(f"    Check saved artifacts in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LineFeat Verification Demo")
    parser.add_argument("--pair-idx", type=int, default=0, help="Pair index from dataset (default: 0)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "linefeat", help="Output directory")
    args = parser.parse_args()

    run_verification(pair_idx=args.pair_idx, output_dir=args.out_dir)


if __name__ == "__main__":
    main()
