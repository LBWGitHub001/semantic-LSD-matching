"""
Example: How to call the trained Line-Transformer in Python.

This script demonstrates:
  1. Simplest 3-line call with image file paths
  2. Accessing matched line coordinates and similarity scores
  3. Generating and saving side-by-side matching visualization
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import LineMatchPipeline, LineFeatPairDataset


def main() -> None:
    print("=================================================================")
    print("           LineFeat High-Level API Calling Example               ")
    print("=================================================================")

    # 1. Initialize Pipeline (automatically loads best_model.pth on GPU)
    pipeline = LineMatchPipeline(
        weights="checkpoints/linefeat/best_model.pth",
        min_line_length=25.0,
        similarity_threshold=0.85,
    )
    print(f"[*] Initialized LineMatchPipeline on device: {pipeline.device}")

    # 2. Test images (using the curated test samples)
    img_path0 = ROOT / "data" / "test_samples" / "pair1_small_rotation" / "image0.png"
    img_path1 = ROOT / "data" / "test_samples" / "pair1_small_rotation" / "image1.png"

    # Fallback to dataset if test_samples not yet present
    if not img_path0.exists():
        ds = LineFeatPairDataset(split="val", min_matches=5)
        sample = ds[0]
        img_path0 = Path(sample["image0_path"])
        img_path1 = Path(sample["image1_path"])

    print(f"\n[1] Matching two images:")
    print(f"    Image 0: {img_path0}")
    print(f"    Image 1: {img_path1}")

    # 3. One-line Matching Call!
    result = pipeline.match(img_path0, img_path1)

    # 4. Inspect Results
    print(f"\n[2] Matching Results:")
    print(f"    Detected Lines: View 0 = {len(result.lines0)}, View 1 = {len(result.lines1)}")
    print(f"    Matched Pairs:  {len(result)} lines successfully matched!")

    if len(result) > 0:
        print("\n    Top 5 matches:")
        for k in range(min(5, len(result))):
            idx0, idx1, score = result.matches[k]
            l0 = result.matched_lines0[k]  # [x1, y1, x2, y2]
            l1 = result.matched_lines1[k]
            print(f"      #{k+1}: Line {idx0:3d} <-> Line {idx1:3d} | Score: {score:.3f}")
            print(f"          View 0: ({l0[0]:.1f}, {l0[1]:.1f}) -> ({l0[2]:.1f}, {l0[3]:.1f})")
            print(f"          View 1: ({l1[0]:.1f}, {l1[1]:.1f}) -> ({l1[2]:.1f}, {l1[3]:.1f})")

    # 5. One-line Visualization Save!
    save_path = "outputs/linefeat/example_call_result.jpg"
    result.save_visualization(save_path)
    print(f"\n[3] Saved visualization image to: {save_path}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
