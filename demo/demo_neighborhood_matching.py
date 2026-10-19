"""
Experiment: Line Neighborhood Transformation & Warped Patch Matching.

Tests the user's hypothesis:
  "Transform a line's neighborhood and itself to align near another line,
   and design a descriptor to judge whether the two are similar enough for matching."

Evaluates:
  1. Canonical Rectified Line Ribbon Extraction (ZNCC)
  2. Direct 2D Line Neighborhood Warping (Affinely mapping Image 0 patch onto Image 1)
  3. Deep Convolutional Strip Descriptor Correlation
  4. Separation power between True Matches and False Matches
  5. Multi-panel visual verification saved to outputs/linefeat/neighborhood_alignment_experiment.jpg
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
import torchvision.models as models

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import (
    LineFeatPairDataset,
    LineNeighborhoodTransformer,
    compute_strip_zncc,
    extract_deep_strip_descriptor,
)


class ResNetBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
        return self.layer2(x)


def create_line_patch_visualization(
    img0: np.ndarray,
    img1: np.ndarray,
    line0: np.ndarray,
    line1: np.ndarray,
    transformer: LineNeighborhoodTransformer,
    title: str,
    zncc_score: float,
    deep_sim: float,
) -> np.ndarray:
    """
    Creates a visual breakdown panel showing:
      [Line in Img0] | [Line in Img1] | [Strip 0] | [Strip 1] | [Warped Patch] | [Difference Map]
    """
    # 1. Extract canonical strips
    strip0 = transformer.extract_canonical_strip(img0, line0)
    strip1 = transformer.extract_canonical_strip(img1, line1)

    # 2. Warp line 0's neighborhood directly to line 1 in Image 1
    warped0, _ = transformer.warp_line1_to_line2(img0, line0, line1, output_shape=img1.shape[:2])

    # Crop a region of interest around line 1
    x1, y1, x2, y2 = [int(round(v)) for v in line1]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    crop_size = 180
    h, w = img1.shape[:2]
    ymin, ymax = max(0, cy - crop_size), min(h, cy + crop_size)
    xmin, xmax = max(0, cx - crop_size), min(w, cx + crop_size)

    crop_img1 = img1[ymin:ymax, xmin:xmax].copy()
    crop_warped0 = warped0[ymin:ymax, xmin:xmax].copy()

    # Draw line on crop_img1
    cv2.line(crop_img1, (x1 - xmin, y1 - ymin), (x2 - xmin, y2 - ymin), (0, 255, 0), 2, cv2.LINE_AA)

    # Compute difference heatmap
    diff = cv2.absdiff(crop_img1, crop_warped0)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    # Crop around line 0 in image 0
    x1_0, y1_0, x2_0, y2_0 = [int(round(v)) for v in line0]
    cx0, cy0 = (x1_0 + x2_0) // 2, (y1_0 + y2_0) // 2
    ymin0, ymax0 = max(0, cy0 - crop_size), min(h, cy0 + crop_size)
    xmin0, xmax0 = max(0, cx0 - crop_size), min(w, cx0 + crop_size)
    crop_img0 = img0[ymin0:ymax0, xmin0:xmax0].copy()
    cv2.line(crop_img0, (x1_0 - xmin0, y1_0 - ymin0), (x2_0 - xmin0, y2_0 - ymin0), (0, 0, 255), 2, cv2.LINE_AA)

    # Standardize crop sizes to (240, 240)
    def resize_box(im: np.ndarray, size: int = 220) -> np.ndarray:
        return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)

    box_img0 = resize_box(crop_img0)
    box_img1 = resize_box(crop_img1)
    box_warped = resize_box(crop_warped0)
    box_diff = resize_box(diff_heatmap)

    # Annotate boxes
    cv2.putText(box_img0, "1. Line 0 (View 0)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(box_img1, "2. Line 1 (View 1)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(box_warped, "3. Warped 0 -> 1", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(box_diff, "4. Absolute Diff", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Resize strips to height=48, width=220*2+10
    strip_w_show = 445
    strip0_show = cv2.resize(strip0, (strip_w_show, 45), interpolation=cv2.INTER_NEAREST)
    strip1_show = cv2.resize(strip1, (strip_w_show, 45), interpolation=cv2.INTER_NEAREST)
    cv2.putText(strip0_show, "Rectified Ribbon 0", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(strip1_show, "Rectified Ribbon 1", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Upper row: 4 boxes
    upper = np.hstack([box_img0, box_img1, box_warped, box_diff])
    # Lower row: 2 stacked strips aligned with upper row width
    lower = np.vstack([strip0_show, strip1_show])
    lower_padded = cv2.resize(lower, (upper.shape[1], 100), interpolation=cv2.INTER_NEAREST)

    panel = np.vstack([upper, lower_padded])

    # Header bar
    header = np.zeros((45, panel.shape[1], 3), dtype=np.uint8)
    color = (0, 255, 0) if "TRUE" in title else (0, 0, 255)
    text = f"{title} | ZNCC Score: {zncc_score:.3f} | Deep Feat Cosine Sim: {deep_sim:.3f}"
    cv2.putText(header, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    return np.vstack([header, panel])


def run_experiment() -> None:
    print("[*] Starting Line Neighborhood Transformation & Matching Experiment...")
    out_dir = ROOT / "outputs" / "linefeat"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset pair
    dataset = LineFeatPairDataset(
        index_path="data/line_gt_dataset/index.json",
        min_matches=5,
        load_images=True,
        load_semantics=True,
    )
    sample = dataset[0]  # scene_00000_485142_0_2
    pair_id = sample["pair_id"]
    gt_lines0 = sample["gt_lines0"].numpy()
    gt_lines1 = sample["gt_lines1"].numpy()
    num_gt = len(gt_lines0)
    print(f"    Loaded Pair: {pair_id} with {num_gt} true matching line pairs.")

    img0_t = sample["image0"]
    img1_t = sample["image1"]
    h, w = img0_t.shape[1], img0_t.shape[2]

    img0_bgr = (img0_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()
    img1_bgr = (img1_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()

    # 2. Extract deep feature maps
    print("    Extracting deep feature maps (ResNet50)...")
    backbone = ResNetBackbone().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t0 = (img0_t.unsqueeze(0).to(device) - mean) / std
    t1 = (img1_t.unsqueeze(0).to(device) - mean) / std
    with torch.no_grad():
        feat0 = backbone(t0)
        feat1 = backbone(t1)

    transformer = LineNeighborhoodTransformer(strip_width=128, strip_height=32, fixed_band_pixels=25.0)

    # 3. Test on True Matches (Positive Pairs)
    pos_zncc_scores: list[float] = []
    pos_deep_sims: list[float] = []

    for i in range(num_gt):
        l0 = gt_lines0[i]
        l1 = gt_lines1[i]

        strip0 = transformer.extract_canonical_strip(img0_bgr, l0)
        strip1 = transformer.extract_canonical_strip(img1_bgr, l1)
        zncc = compute_strip_zncc(strip0, strip1, check_both_orientations=True)
        pos_zncc_scores.append(zncc)

        desc0 = extract_deep_strip_descriptor(feat0, l0, orig_size=(h, w), band_pixels=25.0)
        desc1 = extract_deep_strip_descriptor(feat1, l1, orig_size=(h, w), band_pixels=25.0)
        sim = float(torch.dot(desc0, desc1).item())
        pos_deep_sims.append(sim)

    # 4. Test on False Matches (Negative Pairs - Permuting lines)
    neg_zncc_scores: list[float] = []
    neg_deep_sims: list[float] = []

    np.random.seed(42)
    num_neg = min(50, num_gt * (num_gt - 1))
    neg_indices: list[tuple[int, int]] = []
    while len(neg_indices) < num_neg:
        i = np.random.randint(0, num_gt)
        j = np.random.randint(0, num_gt)
        if i != j and (i, j) not in neg_indices:
            neg_indices.append((i, j))

    for (i, j) in neg_indices:
        l0 = gt_lines0[i]
        l1 = gt_lines1[j]

        strip0 = transformer.extract_canonical_strip(img0_bgr, l0)
        strip1 = transformer.extract_canonical_strip(img1_bgr, l1)
        zncc = compute_strip_zncc(strip0, strip1, check_both_orientations=True)
        neg_zncc_scores.append(zncc)

        desc0 = extract_deep_strip_descriptor(feat0, l0, orig_size=(h, w), band_pixels=25.0)
        desc1 = extract_deep_strip_descriptor(feat1, l1, orig_size=(h, w), band_pixels=25.0)
        sim = float(torch.dot(desc0, desc1).item())
        neg_deep_sims.append(sim)

    # 5. Statistical Comparison
    mean_pos_zncc = np.mean(pos_zncc_scores)
    std_pos_zncc = np.std(pos_zncc_scores)
    mean_neg_zncc = np.mean(neg_zncc_scores)
    std_neg_zncc = np.std(neg_zncc_scores)

    mean_pos_deep = np.mean(pos_deep_sims)
    std_pos_deep = np.std(pos_deep_sims)
    mean_neg_deep = np.mean(neg_deep_sims)
    std_neg_deep = np.std(neg_deep_sims)

    print("\n=================================================================")
    print("           LINE NEIGHBORHOOD ALIGNMENT EXPERIMENTAL RESULTS       ")
    print("=================================================================")
    print("Method 1: Normalized Cross Correlation (ZNCC) on Warped Ribbon Strips")
    print(f"  True Matches (Positive):  Mean = {mean_pos_zncc:.3f} +/- {std_pos_zncc:.3f}  [Min: {np.min(pos_zncc_scores):.3f}, Max: {np.max(pos_zncc_scores):.3f}]")
    print(f"  False Matches (Negative): Mean = {mean_neg_zncc:.3f} +/- {std_neg_zncc:.3f}  [Min: {np.min(neg_zncc_scores):.3f}, Max: {np.max(neg_zncc_scores):.3f}]")
    zncc_margin = mean_pos_zncc - mean_neg_zncc
    print(f"  Separation Margin:        {zncc_margin:+.3f}")

    print("\nMethod 2: Deep Convolutional Strip Descriptor Cosine Similarity")
    print(f"  True Matches (Positive):  Mean = {mean_pos_deep:.3f} +/- {std_pos_deep:.3f}  [Min: {np.min(pos_deep_sims):.3f}, Max: {np.max(pos_deep_sims):.3f}]")
    print(f"  False Matches (Negative): Mean = {mean_neg_deep:.3f} +/- {std_neg_deep:.3f}  [Min: {np.min(neg_deep_sims):.3f}, Max: {np.max(neg_deep_sims):.3f}]")
    deep_margin = mean_pos_deep - mean_neg_deep
    print(f"  Separation Margin:        {deep_margin:+.3f}")
    print("=================================================================\n")

    # 6. Generate Side-by-Side Visual Comparison Panels
    # Pick best positive pair and a typical negative pair
    best_pos_idx = int(np.argmax(pos_zncc_scores))
    worst_neg_idx = 0  # Pick first negative pair

    panel_pos = create_line_patch_visualization(
        img0_bgr,
        img1_bgr,
        gt_lines0[best_pos_idx],
        gt_lines1[best_pos_idx],
        transformer,
        title=f"TRUE MATCH (Line ID {sample['gt_line_ids'][best_pos_idx].item()})",
        zncc_score=pos_zncc_scores[best_pos_idx],
        deep_sim=pos_deep_sims[best_pos_idx],
    )

    neg_i, neg_j = neg_indices[worst_neg_idx]
    panel_neg = create_line_patch_visualization(
        img0_bgr,
        img1_bgr,
        gt_lines0[neg_i],
        gt_lines1[neg_j],
        transformer,
        title=f"FALSE MATCH (Mismatch Line {sample['gt_line_ids'][neg_i].item()} vs {sample['gt_line_ids'][neg_j].item()})",
        zncc_score=neg_zncc_scores[worst_neg_idx],
        deep_sim=neg_deep_sims[worst_neg_idx],
    )

    # Combine both panels
    divider = np.zeros((15, panel_pos.shape[1], 3), dtype=np.uint8)
    combined = np.vstack([panel_pos, divider, panel_neg])

    save_path = out_dir / "neighborhood_alignment_experiment.jpg"
    cv2.imwrite(str(save_path), combined)
    print(f"[+] Multi-panel visual verification saved to: {save_path}")


if __name__ == "__main__":
    run_experiment()
