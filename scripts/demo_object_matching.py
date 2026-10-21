"""
Object Matching ("连连看") Quick Inference Demo.

Run inference on any view pair using trained XFeatObjectMatcher checkpoint:
  1. Loads RGB images and object instance masks.
  2. Runs forward pass through XFeatObjectMatcher.
  3. Outputs matched object pairs and unmatched objects.
  4. Exports side-by-side visualization with connection lines.

Usage:
  # Run on a scene pair:
  python scripts/demo_object_matching.py --scene data/synthetic_object_matching_scenes/scene_0045 --v0 0 --v1 2

  # Custom output image:
  python scripts/demo_object_matching.py --scene data/synthetic_object_matching_scenes/scene_0045 --v0 0 --v1 4 --out demo_match.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat.xfeat_object_matcher import XFeatObjectMatcher
from scripts.train_object_matching import predict_bipartite_matches, visualize_object_matches


def run_inference(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    objs0: list[dict],
    objs1: list[dict],
    checkpoint_path: str = "checkpoints/xfeat_object_matcher/best_model.pth",
    device_name: str = "cuda",
) -> tuple[list[tuple[int, int]], list[int], list[int], np.ndarray]:
    """
    Core Python API for object bipartite matching inference.
    """
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    # 1. Load Model
    model = XFeatObjectMatcher().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 2. Normalize RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    rgb0 = ((img0_bgr[..., ::-1] / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)
    rgb1 = ((img1_bgr[..., ::-1] / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)

    # 3. Build Mask and BBox Tensors
    m0_list, b0_list = [], []
    for obj in objs0:
        m0_list.append((mask0 == obj["inst_id"]).astype(np.float32))
        b0_list.append(obj["bbox"])

    m1_list, b1_list = [], []
    for obj in objs1:
        m1_list.append((mask1 == obj["inst_id"]).astype(np.float32))
        b1_list.append(obj["bbox"])

    t_img0 = torch.from_numpy(rgb0).unsqueeze(0).to(device)
    t_img1 = torch.from_numpy(rgb1).unsqueeze(0).to(device)
    t_m0 = torch.from_numpy(np.stack(m0_list, axis=0)).unsqueeze(0).to(device)
    t_b0 = torch.from_numpy(np.array(b0_list, dtype=np.float32)).unsqueeze(0).to(device)
    t_m1 = torch.from_numpy(np.stack(m1_list, axis=0)).unsqueeze(0).to(device)
    t_b1 = torch.from_numpy(np.array(b1_list, dtype=np.float32)).unsqueeze(0).to(device)

    # 4. Forward
    with torch.no_grad():
        out = model(t_img0, t_img1, t_m0, t_b0, t_m1, t_b1)
        prob_matrix = out["prob_matrix"][0]  # (M0+1, M1+1)

    matches, un0, un1 = predict_bipartite_matches(prob_matrix, thresh=0.2)
    return matches, un0, un1, prob_matrix.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Object Matching Quick Demo")
    parser.add_argument("--scene", type=str, default="data/synthetic_object_matching_scenes/scene_0045", help="Scene directory")
    parser.add_argument("--v0", type=int, default=0, help="View 0 index")
    parser.add_argument("--v1", type=int, default=2, help="View 1 index")
    parser.add_argument("--ckpt", type=str, default="checkpoints/xfeat_object_matcher/best_model.pth", help="Checkpoint path")
    parser.add_argument("--out", type=str, default="outputs/demo_match_result.jpg", help="Output visualization path")
    args = parser.parse_args()

    s_dir = Path(args.scene)
    if not s_dir.exists():
        raise FileNotFoundError(f"Scene directory {s_dir} not found.")

    # Load data
    img0 = cv2.imread(str(s_dir / f"view_{args.v0:02d}.png"))
    img1 = cv2.imread(str(s_dir / f"view_{args.v1:02d}.png"))
    mask0 = cv2.imread(str(s_dir / f"instance_mask_{args.v0:02d}.png"), cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.imread(str(s_dir / f"instance_mask_{args.v1:02d}.png"), cv2.IMREAD_GRAYSCALE)

    with open(s_dir / f"objects_{args.v0:02d}.json", "r", encoding="utf-8") as f:
        objs0 = json.load(f)
    with open(s_dir / f"objects_{args.v1:02d}.json", "r", encoding="utf-8") as f:
        objs1 = json.load(f)

    # Ground truth (if available)
    gt_file = s_dir / "pair_matches.json"
    gt_matches = []
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            pair_data = json.load(f)
        for p in pair_data["pairs"]:
            if (p["view_i"] == args.v0 and p["view_j"] == args.v1) or (p["view_i"] == args.v1 and p["view_j"] == args.v0):
                gt_matches = [tuple(m) if p["view_i"] == args.v0 else (m[1], m[0]) for m in p["gt_matches"]]
                break

    print("=" * 70)
    print("  XFeatObjectMatcher: Cross-View '连连看' Inference Demo")
    print("=" * 70)
    print(f"[*] Scene: {s_dir.name} | View {args.v0} <--> View {args.v1}")
    print(f"[*] View {args.v0} Objects: {[o['category'] for o in objs0]}")
    print(f"[*] View {args.v1} Objects: {[o['category'] for o in objs1]}")

    # Run inference
    matches, un0, un1, P = run_inference(img0, img1, mask0, mask1, objs0, objs1, checkpoint_path=args.ckpt)

    print("\n[+] MATCHING RESULTS:")
    for i, j in matches:
        cat0 = objs0[i]["category"]
        cat1 = objs1[j]["category"]
        score = P[i, j]
        is_correct = (i, j) in gt_matches if gt_matches else True
        tag = "✓ TRUE" if is_correct else "✗ FALSE"
        print(f"  • View0 [{cat0} #{i}] <===> View1 [{cat1} #{j}] | Prob: {score:.3f} | {tag}")

    if un0:
        print(f"\n[-] UNMATCHED in View {args.v0} (Dustbin): {[objs0[i]['category'] for i in un0]}")
    if un1:
        print(f"[-] UNMATCHED in View {args.v1} (Dustbin): {[objs1[j]['category'] for j in un1]}")

    # Visualize
    out_path = Path(args.out)
    visualize_object_matches(
        img0, img1, objs0, objs1, matches, gt_matches, un0, un1, out_path,
        title=f"XFeatObjectMatcher: {s_dir.name} (View {args.v0} <-> View {args.v1})",
    )
    print(f"\n[✓] Visualization exported to -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
