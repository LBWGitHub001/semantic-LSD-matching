#!/usr/bin/env python3
"""
Structured3D Real-World Object Matching Dataset Builder.

Processes perspective views in Structured3D_perspective_full_00:
  1. Extracts individual 2D furniture/object instance masks & bounding boxes from semantic.png via connected components.
  2. Back-projects mask pixels to 3D world space using depth.png and camera_pose.txt to compute invariant 3D centroids.
  3. Computes cross-view bipartite ground truth (1-to-1 matching pairs and out-of-view Dustbin objects).
  4. Exports into unified ObjectMatchingDataset format (RGB images, instance masks, objects JSON, pair_matches JSON).

Usage:
  python tools/build_structured3d_object_dataset.py
  python tools/build_structured3d_object_dataset.py --max-scenes 50 --num-workers 16
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.read_structured3d import Camera, STRUCTURED3D_CLASSES

TARGET_CATEGORIES = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "desk", "wardrobe", "television",
    "nightstand", "refrigerator", "sink", "lamp",
]


def extract_view_objects(
    view_dir: Path,
    target_h: int = 480,
    target_w: int = 640,
    min_pixel_area: int = 300,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Extracts object instances, 2D masks, and 3D world centroids from a single Structured3D view.

    Returns:
      rgb_resized: (target_h, target_w, 3) BGR image
      mask_resized: (target_h, target_w) uint8 instance mask
      objects: list of object dictionaries
    """
    cam_file = view_dir / "camera_pose.txt"
    rgb_file = view_dir / "rgb_rawlight.png"
    depth_file = view_dir / "depth.png"
    sem_file = view_dir / "semantic.png"

    cam = Camera.from_file(cam_file)
    bgr = cv2.imread(str(rgb_file))
    if bgr is None:
        raise FileNotFoundError(f"Missing image {rgb_file}")

    depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Missing depth {depth_file}")
    depth = depth_raw.astype(np.float32) / 1000.0  # mm to meters

    sem_raw = cv2.imread(str(sem_file), cv2.IMREAD_UNCHANGED)
    if sem_raw is None:
        raise FileNotFoundError(f"Missing semantic {sem_file}")
    sem = sem_raw[..., 0] if sem_raw.ndim == 3 else sem_raw

    H_orig, W_orig = sem.shape
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    R_c2w = cam.R_c2w
    t_w = cam.position / 1000.0  # Camera origin in meters

    inst_mask_orig = np.zeros((H_orig, W_orig), dtype=np.uint8)
    objects = []
    inst_id = 1

    for cat in TARGET_CATEGORIES:
        if cat not in STRUCTURED3D_CLASSES:
            continue
        cls_id = STRUCTURED3D_CLASSES.index(cat)
        bin_mask = (sem == cls_id).astype(np.uint8)
        if bin_mask.sum() < min_pixel_area:
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
        for l in range(1, num_labels):
            area = stats[l, cv2.CC_STAT_AREA]
            if area < min_pixel_area:
                continue

            x0, y0, w, h = stats[l, cv2.CC_STAT_LEFT], stats[l, cv2.CC_STAT_TOP], stats[l, cv2.CC_STAT_WIDTH], stats[l, cv2.CC_STAT_HEIGHT]
            mask_l = (labels == l)

            ys, xs = np.where(mask_l)
            zs = depth[ys, xs]
            valid = (zs > 0.2) & (zs < 12.0)
            if valid.sum() < 100:
                continue

            xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
            # Subsample points for fast centroid computation
            step = max(1, len(xs_v) // 250)
            xs_s, ys_s, zs_s = xs_v[::step], ys_v[::step], zs_v[::step]

            # Back-project to camera coordinates
            xc = (xs_s - cx) * zs_s / fx
            yc = (ys_s - cy) * zs_s / fy
            zc = zs_s
            pts_c = np.stack([xc, yc, zc], axis=1)

            # Transform to world coordinates
            pts_w = pts_c @ R_c2w.T + t_w
            c_w = np.median(pts_w, axis=0)

            # Assign instance mask
            inst_mask_orig[mask_l] = inst_id

            # Scale bounding box to target resolution
            scale_x = target_w / float(W_orig)
            scale_y = target_h / float(H_orig)
            bbox_scaled = [
                round(x0 * scale_x, 1),
                round(y0 * scale_y, 1),
                round((x0 + w) * scale_x, 1),
                round((y0 + h) * scale_y, 1),
            ]

            cx_2d = round((bbox_scaled[0] + bbox_scaled[2]) / 2.0, 1)
            cy_2d = round((bbox_scaled[1] + bbox_scaled[3]) / 2.0, 1)
            objects.append({
                "inst_id": inst_id,
                "category": cat,
                "bbox": bbox_scaled,
                "centroid": [cx_2d, cy_2d],
                "centroid_3d": [round(float(c), 3) for c in c_w],
                "pixel_count": int(area),
            })
            inst_id += 1

    # Resize image and instance mask to standard training size
    rgb_resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(inst_mask_orig, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return rgb_resized, mask_resized, objects


def associate_view_pair(
    objs0: list[dict[str, Any]],
    objs1: list[dict[str, Any]],
    max_centroid_dist: float = 0.40,
) -> tuple[list[list[int]], list[int], list[int]]:
    """
    Computes bipartite ground truth matches and dustbin elements via 3D spatial mutual nearest neighbors.
    """
    M0, M1 = len(objs0), len(objs1)
    if M0 == 0 or M1 == 0:
        return [], list(range(M0)), list(range(M1))

    # Compute pairwise distance matrix for objects of matching categories
    dist_matrix = np.full((M0, M1), fill_value=1e6, dtype=np.float32)
    for i, o0 in enumerate(objs0):
        c0 = np.array(o0["centroid_3d"], dtype=np.float32)
        for j, o1 in enumerate(objs1):
            if o0["category"] == o1["category"]:
                c1 = np.array(o1["centroid_3d"], dtype=np.float32)
                d = float(np.linalg.norm(c0 - c1))
                if d <= max_centroid_dist:
                    dist_matrix[i, j] = d

    # Mutual Nearest Neighbor (MNN) bipartite matching
    gt_matches = []
    matched_0, matched_1 = set(), set()

    for i in range(M0):
        min_j = int(np.argmin(dist_matrix[i, :]))
        if dist_matrix[i, min_j] <= max_centroid_dist:
            min_i = int(np.argmin(dist_matrix[:, min_j]))
            if min_i == i and min_j not in matched_1:
                gt_matches.append([i, min_j])
                matched_0.add(i)
                matched_1.add(min_j)

    unmatched_0 = [i for i in range(M0) if i not in matched_0]
    unmatched_1 = [j for j in range(M1) if j not in matched_1]

    return gt_matches, unmatched_0, unmatched_1


def process_scene(
    scene_dir_str: str,
    output_dir_str: str,
    target_h: int = 480,
    target_w: int = 640,
    min_matches: int = 2,
    max_rotation_deg: float = 90.0,
) -> dict[str, Any]:
    """Processes all valid room pairs in a single Structured3D scene."""
    scene_dir = Path(scene_dir_str)
    output_dir = Path(output_dir_str)
    scene_id = scene_dir.name

    room_dirs = sorted([d for d in (scene_dir / "2D_rendering").iterdir() if d.is_dir()])
    scene_out_dir = output_dir / scene_id
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    extracted_views = {}
    pair_records = []
    pair_counter = 0

    for room in room_dirs:
        persp_dir = room / "perspective" / "full"
        if not persp_dir.is_dir():
            continue

        view_dirs = sorted([v for v in persp_dir.iterdir() if v.is_dir() and (v / "camera_pose.txt").is_file()])
        if len(view_dirs) < 2:
            continue

        # Extract objects for each view in this room
        room_views = {}
        for vd in view_dirs:
            v_key = f"{room.name}_{vd.name}"
            try:
                rgb_r, mask_r, objs = extract_view_objects(vd, target_h, target_w)
                room_views[v_key] = {
                    "rgb": rgb_r,
                    "mask": mask_r,
                    "objs": objs,
                    "cam": Camera.from_file(vd / "camera_pose.txt"),
                }
            except Exception:
                continue

        # Form pairs within same room
        keys = list(room_views.keys())
        for a_idx in range(len(keys)):
            for b_idx in range(a_idx + 1, len(keys)):
                ka, kb = keys[a_idx], keys[b_idx]
                va, vb = room_views[ka], room_views[kb]

                cam_a, cam_b = va["cam"], vb["cam"]
                rot_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(cam_a.R_c2w.T @ cam_b.R_c2w) - 1.0) / 2.0, -1.0, 1.0))))
                base_m = float(np.linalg.norm(cam_a.position - cam_b.position) / 1000.0)

                if rot_deg > max_rotation_deg or (rot_deg < 10.0 and base_m < 0.3):
                    continue

                gt_matches, un0, un1 = associate_view_pair(va["objs"], vb["objs"])
                if len(gt_matches) < min_matches:
                    continue

                # Save pairs
                v0_idx = pair_counter * 2
                v1_idx = pair_counter * 2 + 1

                cv2.imwrite(str(scene_out_dir / f"view_{v0_idx:02d}.png"), va["rgb"])
                cv2.imwrite(str(scene_out_dir / f"view_{v1_idx:02d}.png"), vb["rgb"])
                cv2.imwrite(str(scene_out_dir / f"instance_mask_{v0_idx:02d}.png"), va["mask"])
                cv2.imwrite(str(scene_out_dir / f"instance_mask_{v1_idx:02d}.png"), vb["mask"])

                with open(scene_out_dir / f"objects_{v0_idx:02d}.json", "w", encoding="utf-8") as f:
                    json.dump(va["objs"], f, indent=2)
                with open(scene_out_dir / f"objects_{v1_idx:02d}.json", "w", encoding="utf-8") as f:
                    json.dump(vb["objs"], f, indent=2)

                pair_records.append({
                    "view_i": v0_idx,
                    "view_j": v1_idx,
                    "num_matches": len(gt_matches),
                    "gt_matches": gt_matches,
                    "unmatched_i": un0,
                    "unmatched_j": un1,
                    "rotation_deg": round(rot_deg, 2),
                    "baseline_m": round(base_m, 2),
                    "room_id": room.name,
                })
                pair_counter += 1

    if pair_records:
        with open(scene_out_dir / "pair_matches.json", "w", encoding="utf-8") as f:
            json.dump({
                "scene_id": scene_id,
                "num_pairs": len(pair_records),
                "pairs": pair_records,
            }, f, indent=2)

    return {"scene_id": scene_id, "pairs_created": len(pair_records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured3D Real-World Object Matching Dataset Builder")
    parser.add_argument("--s3d-dir", type=str, default="Structured3D_perspective_full_00/Structured3D", help="Structured3D raw directory")
    parser.add_argument("--out-dir", type=str, default="data/structured3d_object_matching", help="Output dataset directory")
    parser.add_argument("--max-scenes", type=int, default=50, help="Max scenes to process (default: 50)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 16, help="Parallel worker processes")
    parser.add_argument("--min-matches", type=int, default=2, help="Minimum matches per pair")
    parser.add_argument("--max-rotation", type=float, default=85.0, help="Maximum camera rotation in degrees")
    args = parser.parse_args()

    s3d_dir = Path(args.s3d_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted([d for d in s3d_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")])
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    print("=" * 70)
    print("      Structured3D Real-World Object Matching Dataset Builder      ")
    print("=" * 70)
    print(f"  Source Directory: {s3d_dir.resolve()}")
    print(f"  Output Directory: {out_dir.resolve()}")
    print(f"  Target Scenes:    {len(scenes)}")
    print(f"  Worker Processes: {args.num_workers}")
    print("=" * 70 + "\n")

    t0 = time.time()
    valid_scenes = []
    total_pairs = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_scene, str(s), str(out_dir),
                min_matches=args.min_matches, max_rotation_deg=args.max_rotation
            ): s.name for s in scenes
        }

        for idx, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res["pairs_created"] > 0:
                valid_scenes.append(res["scene_id"])
                total_pairs += res["pairs_created"]
            print(f"  [{idx+1:3d}/{len(scenes):3d}] Scene {res['scene_id']} -> {res['pairs_created']} valid pairs created.")

    # Sort valid scenes
    valid_scenes.sort()
    n_total = len(valid_scenes)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_scenes = valid_scenes[:n_train]
    val_scenes = valid_scenes[n_train:n_train+n_val]
    test_scenes = valid_scenes[n_train+n_val:]

    with open(out_dir / "train_split.json", "w", encoding="utf-8") as f:
        json.dump(train_scenes, f, indent=2)
    with open(out_dir / "val_split.json", "w", encoding="utf-8") as f:
        json.dump(val_scenes, f, indent=2)
    with open(out_dir / "test_split.json", "w", encoding="utf-8") as f:
        json.dump(test_scenes, f, indent=2)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  [✓] DATASET GENERATION COMPLETE!")
    print(f"  • Total Valid Scenes: {n_total} (Train: {len(train_scenes)}, Val: {len(val_scenes)}, Test: {len(test_scenes)})")
    print(f"  • Total Valid Pairs:  {total_pairs}")
    print(f"  • Time Elapsed:       {elapsed:.1f}s ({total_pairs / max(elapsed, 1e-3):.1f} pairs/sec)")
    print(f"  • Splits Exported:    train_split.json, val_split.json, test_split.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
