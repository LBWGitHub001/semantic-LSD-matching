"""
Unified Interactive Inference Demo & Visualizer.

Usage:
  # 1. Run demo on synthetic multi-view scene pair:
  python demo.py --model xfeat_object --scene data/synthetic_object_matching_scenes/scene_0045 --v0 0 --v1 3

  # 2. Specify custom checkpoint and output file:
  python demo.py --model xfeat_object --checkpoint checkpoints/xfeat_object_matcher/best_model.pth --out outputs/demo.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model
from datasets import predict_bipartite_matches, visualize_object_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Interactive Matching Demo")
    parser.add_argument("--model", type=str, default="xfeat_object",
                        choices=["xfeat_object", "xfeat", "mask_line", "line_transformer"],
                        help="Model architecture name")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/xfeat_object_matcher/best_model.pth",
                        help="Checkpoint weights path")
    parser.add_argument("--scene", type=str, default="data/synthetic_object_matching_scenes/scene_0045",
                        help="Path to scene directory")
    parser.add_argument("--v0", type=int, default=0, help="View 0 index")
    parser.add_argument("--v1", type=int, default=3, help="View 1 index")
    parser.add_argument("--thresh", type=float, default=0.2, help="Matching probability threshold")
    parser.add_argument("--out", type=str, default="outputs/demo_result.jpg", help="Output visualization image path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_object_demo(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    s_dir = Path(args.scene)
    if not s_dir.exists():
        raise FileNotFoundError(f"Scene directory {s_dir} not found.")

    img0 = cv2.imread(str(s_dir / f"view_{args.v0:02d}.png"))
    img1 = cv2.imread(str(s_dir / f"view_{args.v1:02d}.png"))
    mask0 = cv2.imread(str(s_dir / f"instance_mask_{args.v0:02d}.png"), cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.imread(str(s_dir / f"instance_mask_{args.v1:02d}.png"), cv2.IMREAD_GRAYSCALE)

    with open(s_dir / f"objects_{args.v0:02d}.json", "r", encoding="utf-8") as f:
        objs0 = json.load(f)
    with open(s_dir / f"objects_{args.v1:02d}.json", "r", encoding="utf-8") as f:
        objs1 = json.load(f)

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
    print("      Unified Matching Demo: XFeatObjectMatcher ('连连看')       ")
    print("=" * 70)
    print(f"[*] Scene:          {s_dir.name} (View {args.v0} <--> View {args.v1})")
    print(f"[*] View {args.v0} Objects: {[o['category'] for o in objs0]}")
    print(f"[*] View {args.v1} Objects: {[o['category'] for o in objs1]}")
    print(f"[*] Checkpoint:     {args.checkpoint}")

    # Model
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Normalize RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    rgb0 = ((img0[..., ::-1] / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)
    rgb1 = ((img1[..., ::-1] / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)

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

    with torch.no_grad():
        out = model(t_img0, t_img1, t_m0, t_b0, t_m1, t_b1)
        prob_matrix = out["prob_matrix"][0]

    matches, un0, un1 = predict_bipartite_matches(prob_matrix, thresh=args.thresh)
    P_np = prob_matrix.cpu().numpy()

    print("\n[+] MATCHING RESULTS:")
    for i, j in matches:
        c0, c1 = objs0[i]["category"], objs1[j]["category"]
        score = P_np[i, j]
        is_correct = (i, j) in gt_matches if gt_matches else True
        tag = "✓ TRUE" if is_correct else "✗ FALSE"
        print(f"  • View0 [{c0} #{i}] <===> View1 [{c1} #{j}] | Prob: {score:.3f} | {tag}")

    if un0:
        print(f"\n[-] UNMATCHED in View {args.v0} (Dustbin): {[objs0[i]['category'] for i in un0]}")
    if un1:
        print(f"[-] UNMATCHED in View {args.v1} (Dustbin): {[objs1[j]['category'] for j in un1]}")

    out_path = Path(args.out)
    visualize_object_matches(
        img0, img1, objs0, objs1, matches, gt_matches, un0, un1, out_path,
        title=f"Demo: {s_dir.name} (View {args.v0} <-> View {args.v1})",
    )
    print(f"\n[✓] Visualization exported to -> {out_path.resolve()}")


def run_line_demo(args: argparse.Namespace) -> None:
    from models.line_matcher import LineMatchPipeline

    s_dir = Path(args.scene)
    if not s_dir.exists() or "synthetic_object" in str(s_dir):
        # Auto-switch to multiview scenes if default object scene was kept
        fallback = Path("data/synthetic_multiview_scenes/scene_0000")
        if fallback.exists():
            s_dir = fallback

    views_dir = s_dir / "views" if (s_dir / "views").exists() else s_dir
    img0_path = views_dir / f"view_{args.v0:02d}.png"
    img1_path = views_dir / f"view_{args.v1:02d}.png"
    lines0_path = views_dir / f"lines_{args.v0:02d}.json"
    lines1_path = views_dir / f"lines_{args.v1:02d}.json"

    if not img0_path.exists() or not img1_path.exists():
        raise FileNotFoundError(f"Image views not found in {views_dir}: {img0_path}, {img1_path}")

    lines0 = np.array(json.load(open(lines0_path, "r")), dtype=np.float32) if lines0_path.exists() else np.zeros((0, 4), dtype=np.float32)
    lines1 = np.array(json.load(open(lines1_path, "r")), dtype=np.float32) if lines1_path.exists() else np.zeros((0, 4), dtype=np.float32)

    print("=" * 70)
    print(f"      Unified Matching Demo: Line Matcher ({args.model})       ")
    print("=" * 70)
    print(f"[*] Scene:      {s_dir.name} (View {args.v0} <--> View {args.v1})")
    print(f"[*] Lines 0:    {len(lines0)} line segments")
    print(f"[*] Lines 1:    {len(lines1)} line segments")

    ckpt_path = Path(args.checkpoint) if args.checkpoint and Path(args.checkpoint).is_file() else None
    pipeline = LineMatchPipeline(
        weights=ckpt_path,
        device=args.device,
        model_type=args.model,
    )
    result = pipeline.match(img0_path, img1_path, lines0, lines1, threshold=args.thresh)
    print(f"\n[+] Matched {len(result.matches)} line segments!")
    out_path = Path(args.out)
    result.save_visualization(out_path)
    print(f"[✓] Visualization exported to -> {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    if args.model in ["xfeat_object", "xfeat"]:
        run_object_demo(args)
    else:
        run_line_demo(args)


if __name__ == "__main__":
    main()
