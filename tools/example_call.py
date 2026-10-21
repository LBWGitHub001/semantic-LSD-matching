"""
Minimal Python API Example: Pure Line Matching (Zero Detection).

Input:
  - image0, image1 (Paths or in-memory arrays)
  - lines0 (N0, 4) [x1, y1, x2, y2]
  - lines1 (N1, 4) [x1, y1, x2, y2]

Output:
  - Matched line pairs [(idx0, idx1, similarity_score), ...]
  - Evaluated against ground-truth correspondences from gt_matches.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.line_matcher import LineMatchPipeline


def main() -> None:
    # 1. Select sample pair and mode directory
    pair_dir = ROOT / "data" / "test_samples" / "pair1_small_rotation"
    mode_dir = pair_dir / "mode_a_pure_permuted"

    img0_path = pair_dir / "image0.png"
    img1_path = pair_dir / "image1.png"

    # 2. Directly load 2D lines coordinates from JSON: shape (N, 4)
    with open(mode_dir / "lines0.json", "r") as f:
        lines0 = np.array(json.load(f), dtype=np.float32)

    with open(mode_dir / "lines1.json", "r") as f:
        lines1 = np.array(json.load(f), dtype=np.float32)

    with open(mode_dir / "gt_matches.json", "r") as f:
        gt_matches = json.load(f)

    gt_dict = {m["idx0"]: m["idx1"] for m in gt_matches}

    print("=" * 65)
    print("  LineFeat Pure Matching Example (Default: MaskLineFeatMatchNet)")
    print("=" * 65)
    print(f"[*] Input View 0 Lines: {len(lines0)} lines (given by user/GT)")
    print(f"[*] Input View 1 Lines: {len(lines1)} lines (given by user/GT)")
    print(f"[*] Ground Truth Matches: {len(gt_matches)} true correspondences")

    # 3. Initialize Matcher Pipeline (defaults to MaskLineFeatMatchNet)
    pipeline = LineMatchPipeline("checkpoints/linefeat/best_model.pth", model_type="mask")

    # 4. Pure Matching Call: supply lines0 and lines1 directly!
    #    (Detector is 100% skipped, 6-channel EDT+AFM fields generated automatically)
    result = pipeline.match(
        image0=img0_path,
        image1=img1_path,
        lines0=lines0,
        lines1=lines1,
        threshold=0.70,
    )

    if result.matrix_255 is not None:
        print(f"[*] Canonical Match Matrix Shape: {result.matrix_255.shape} (255x255)")
        active_sub = result.matrix_255[:len(lines0), :len(lines1)]
        print(f"    Active submatrix max prob: {active_sub.max():.4f}, mean prob: {active_sub.mean():.4f}")

    # 5. Inspect and verify against Ground Truth
    print(f"\n[*] Model predicted {len(result)} matches:")
    correct_count = 0
    for rank, (i, j, score) in enumerate(result.matches):
        true_j = gt_dict.get(i)
        is_correct = (true_j == j)
        if is_correct:
            correct_count += 1
        status = "CORRECT" if is_correct else f"WRONG (True is #{true_j})"
        print(f"    Match #{rank+1:2d}: Line0[{i:2d}] <--> Line1[{j:2d}] | Score: {score:.3f} | [{status}]")

    prec = (correct_count / max(len(result), 1)) * 100.0
    recall = (correct_count / max(len(gt_matches), 1)) * 100.0
    print(f"\n[*] Summary Metrics:")
    print(f"    - Precision: {correct_count}/{len(result)} ({prec:.1f}%)")
    print(f"    - Recall:    {correct_count}/{len(gt_matches)} ({recall:.1f}%)")

    # 6. Save visualization image
    out_vis = ROOT / "outputs" / "pure_matching" / "example_call_vis.jpg"
    out_vis.parent.mkdir(parents=True, exist_ok=True)
    result.save_visualization(out_vis)
    print(f"[*] Saved predicted visualization to: {out_vis}")
    print("=" * 65)


if __name__ == "__main__":
    main()
