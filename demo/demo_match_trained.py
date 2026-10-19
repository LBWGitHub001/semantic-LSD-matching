"""
Inference & Visualization Script for Trained Line-Transformer Matcher.

Matches 2D line segments between any pair of images using the trained weights.

Usage:
  # 1. Run on a test pair from the dataset:
  python demo/demo_match_trained.py --pair-idx 0

  # 2. Run on custom user images:
  python demo/demo_match_trained.py path/to/image1.jpg path/to/image2.jpg

  # 3. Specify custom checkpoint:
  python demo/demo_match_trained.py --weights checkpoints/linefeat/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import (
    LineDetector,
    LineMatcher,
    LineMatcherNet,
    LineFeatPairDataset,
    get_distinct_colors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Line Matching with Trained Line-Transformer")
    parser.add_argument("images", nargs="*", type=Path, help="Optional: image1 [image2]. If omitted, loads from dataset.")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints" / "linefeat" / "best_model.pth", help="Trained model checkpoint path")
    parser.add_argument("--pair-idx", type=int, default=0, help="Test pair index when images are omitted (default: 0)")
    parser.add_argument("--min-length", type=float, default=25.0, help="Minimum line length in pixels")
    parser.add_argument("--thresh", type=float, default=0.85, help="Descriptor cosine similarity threshold (default: 0.85)")
    parser.add_argument("--d-model", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--max-draw", type=int, default=50, help="Max matches to draw in visualization")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "linefeat" / "trained_matches.jpg", help="Output visualization path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_image(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """Loads image and returns (bgr_uint8, normalized_tensor)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    return bgr, t


def match_image_pair(
    model: LineMatcherNet,
    detector: LineDetector,
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    img_t0: torch.Tensor,
    img_t1: torch.Tensor,
    device: torch.device,
    min_length: float = 25.0,
    thresh: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    """Detects lines and predicts cross-image matches using LineMatcherNet."""
    # 1. 2D Line Detection via LSD
    lines0 = detector.detect(bgr0, min_length=min_length, margin=8)
    lines1 = detector.detect(bgr1, min_length=min_length, margin=8)

    N0, N1 = len(lines0), len(lines1)
    if N0 == 0 or N1 == 0:
        return lines0, lines1, []

    # 2. ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    t0 = (img_t0.unsqueeze(0).to(device) - mean) / std  # (1, 3, H, W)
    t1 = (img_t1.unsqueeze(0).to(device) - mean) / std

    l0_t = torch.from_numpy(lines0).unsqueeze(0).to(device)  # (1, N0, 4)
    l1_t = torch.from_numpy(lines1).unsqueeze(0).to(device)

    # 3. Model Forward Pass
    with torch.no_grad():
        out = model(t0, t1, l0_t, l1_t)
        scores = out["scores"][0].cpu().numpy()
        log_P = out["log_assignment"][0].cpu().numpy()

    # 4. Extract matches with MNN on Sinkhorn assignment and cosine similarity threshold
    P_mat = log_P[:N0, :N1]  # (N0, N1)
    best1 = np.argmax(P_mat, axis=1)  # (N0,) best in view 1 for each line in view 0
    best0 = np.argmax(P_mat, axis=0)  # (N1,) best in view 0 for each line in view 1

    matches: list[tuple[int, int, float]] = []
    for i in range(N0):
        j = int(best1[i])

        # Mutual nearest neighbor check on assignment matrix
        if int(best0[j]) != i:
            continue

        sim = float(scores[i, j])
        if sim < thresh:
            continue

        matches.append((i, j, sim))

    matches.sort(key=lambda m: m[2], reverse=True)
    return lines0, lines1, matches


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=================================================================")
    print("      Trained Line-Transformer Cross-Image Line Matching        ")
    print("=================================================================")
    print(f"  Device:           {device}")
    print(f"  Weights:          {args.weights}")
    print(f"  Match Threshold:  {args.thresh}")

    # 1. Load Trained Weights
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {args.weights}")

    print(f"\n[1/3] Loading checkpoint: {args.weights.name}...")
    ckpt = torch.load(args.weights, map_location=device)
    epoch = ckpt.get("epoch", "unknown")
    val_acc = ckpt.get("val_acc", 0.0)
    print(f"      Trained Epoch: {epoch} | Best Validation Accuracy: {val_acc:.1f}%")

    model = LineMatcherNet(
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=4,
        freeze_backbone=True,
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    detector = LineDetector(scale=0.8)

    # 2. Prepare Input Images
    gt_lines0, gt_lines1 = None, None
    if len(args.images) >= 2:
        img_path0 = args.images[0]
        img_path1 = args.images[1]
        print(f"\n[2/3] Loading custom images:")
        print(f"      Image 0: {img_path0}")
        print(f"      Image 1: {img_path1}")
        bgr0, img_t0 = load_image(img_path0)
        bgr1, img_t1 = load_image(img_path1)
        source_desc = f"Custom Pair: {img_path0.name} <-> {img_path1.name}"
    else:
        print(f"\n[2/3] Loading test pair from dataset index (pair-idx={args.pair_idx})...")
        ds = LineFeatPairDataset(
            index_path="data/line_gt_dataset/index.json",
            split="val",
            min_matches=5,
            load_images=True,
            load_cameras=False,
        )
        sample = ds[args.pair_idx]
        pair_id = sample["pair_id"]
        img_t0 = sample["image0"]
        img_t1 = sample["image1"]
        gt_lines0 = sample["gt_lines0"].numpy()
        gt_lines1 = sample["gt_lines1"].numpy()
        rot_deg = sample.get("rotation_deg", 0.0)
        baseline = sample.get("baseline_meters", 0.0)

        bgr0 = (img_t0.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()
        bgr1 = (img_t1.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1].copy()
        source_desc = f"Dataset Pair: {pair_id} | Rot: {rot_deg:.1f} deg | Base: {baseline:.2f} m"
        print(f"      Loaded: {source_desc}")

    # 3. Run Matching
    print(f"\n[3/3] Running Line Segment Detection (LSD) & Transformer Matching...")
    lines0, lines1, matches = match_image_pair(
        model, detector, bgr0, bgr1, img_t0, img_t1, device,
        min_length=args.min_length, thresh=args.thresh
    )

    print(f"      Detected Lines: View 0 = {len(lines0)} lines, View 1 = {len(lines1)} lines")
    print(f"      Matches Found:  {len(matches)} pairs (Threshold: {args.thresh})")
    if len(matches) > 0:
        top_scores = [m[2] for m in matches[:10]]
        print(f"      Top-10 Scores:  {[round(s, 3) for s in top_scores]}")

    # 4. Generate Visualization
    canvas = LineMatcher.draw_matches(bgr0, bgr1, lines0, lines1, matches, max_draw=args.max_draw)

    # Custom banner
    banner = np.zeros((35, canvas.shape[1], 3), dtype=np.uint8)
    banner_text = f"Trained Line-Transformer | {source_desc} | {len(matches)} Matches"
    cv2.putText(banner, banner_text, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    final_img = np.vstack([banner, canvas])

    cv2.imwrite(str(args.out), final_img)
    print(f"\n[+] Visualization saved successfully to:")
    print(f"    {args.out}")


if __name__ == "__main__":
    main()
