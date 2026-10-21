#!/usr/bin/env python3
"""
Structured3D Semantic Region Mask Matching Dataset Builder.

Constructs cross-view semantic region correspondence datasets:
  - Each view contains a collection of semantic region binary masks (walls, floors, ceilings, furniture).
  - Cross-view pairs have ground-truth 1-to-1 correspondences between matching semantic regions.
  - Out-of-view / occluded regions are explicitly marked as unmatched (for Dustbin / slack supervision).
  - Exports side-by-side visual correspondence verification images.

Usage:
  python tools/build_semantic_region_dataset.py --max-scenes 50 --num-workers 16
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

from datasets.read_structured3d import Camera, Structured3DView, STRUCTURED3D_CLASSES

FURNITURE_CATEGORIES = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "desk", "wardrobe", "television",
    "nightstand", "sink", "lamp", "counter",
]

DISTINCT_COLORS = [
    (0, 220, 0),     # Bright Green
    (0, 165, 255),   # Orange
    (255, 200, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (50, 100, 255),  # Coral
    (180, 105, 255), # Pink
    (255, 144, 30),  # Sky Blue
    (140, 230, 100), # Light Green
    (80, 240, 240),  # Light Yellow
    (200, 180, 50),  # Teal
    (240, 120, 180), # Purple
]


def extract_plane_masks(
    view: Structured3DView,
    H: int = 480,
    W: int = 640,
    min_pixel_area: int = 400,
) -> list[dict[str, Any]]:
    """Extracts structural planar surface masks from layout.json."""
    layout = view.load_layout()
    junctions = layout.get("junctions", [])
    planes = layout.get("planes", [])
    depth_raw = view.load_depth()
    H_orig, W_orig = depth_raw.shape
    scale_x, scale_y = W / float(W_orig), H / float(H_orig)

    plane_masks = []
    for pl in planes:
        vmask = pl.get("visible_mask", [])
        if not vmask or len(vmask[0]) < 3:
            continue
        poly = []
        for j_idx in vmask[0]:
            if j_idx < len(junctions):
                pt = junctions[j_idx]["coordinate"]
                poly.append([int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))])
        if len(poly) >= 3:
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
            area = int(np.count_nonzero(mask))
            if area >= min_pixel_area:
                pid = pl.get("ID")
                ptype = pl.get("type", "wall")
                ys, xs = np.where(mask > 0)
                plane_masks.append({
                    "entity_id": f"plane_{pid}",
                    "label": f"{ptype}_{pid}",
                    "category": ptype,
                    "region_type": "plane",
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "centroid_2d": [round(float(np.mean(xs)), 1), round(float(np.mean(ys)), 1)],
                    "pixel_area": area,
                    "mask": mask,
                })
    return plane_masks


def extract_furniture_masks(
    view: Structured3DView,
    H: int = 480,
    W: int = 640,
    min_pixel_area: int = 350,
) -> list[dict[str, Any]]:
    """Extracts furniture/object region masks from semantic.png + depth.png."""
    cam = view.camera
    sem_raw = view.load_semantic()
    sem = sem_raw[..., 0] if sem_raw.ndim == 3 else sem_raw
    depth = view.load_depth(in_meters=True)
    H_orig, W_orig = sem.shape
    scale_x, scale_y = W / float(W_orig), H / float(H_orig)

    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    R_c2w = cam.R_c2w
    t_w = cam.position / 1000.0  # Camera origin in meters

    furniture_masks = []
    for cat in FURNITURE_CATEGORIES:
        if cat not in STRUCTURED3D_CLASSES:
            continue
        cls_id = STRUCTURED3D_CLASSES.index(cat)
        bin_m = (sem == cls_id).astype(np.uint8)
        if bin_m.sum() < min_pixel_area:
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_m)
        for l in range(1, num_labels):
            area = stats[l, cv2.CC_STAT_AREA]
            if area < min_pixel_area:
                continue

            mask_l = (labels == l)
            ys, xs = np.where(mask_l)
            zs = depth[ys, xs]
            valid = (zs > 0.2) & (zs < 12.0)
            if valid.sum() < 100:
                continue

            xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
            step = max(1, len(xs_v) // 250)
            pts_c = np.stack([
                (xs_v[::step] - cx) * zs_v[::step] / fx,
                (ys_v[::step] - cy) * zs_v[::step] / fy,
                zs_v[::step]
            ], axis=1)
            pts_w = pts_c @ R_c2w.T + t_w
            c_w = np.median(pts_w, axis=0)

            # Resize mask to target (H, W)
            mask_resized = cv2.resize(mask_l.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
            area_scaled = int(np.count_nonzero(mask_resized))
            if area_scaled < 150:
                continue

            ys_s, xs_s = np.where(mask_resized > 0)
            furniture_masks.append({
                "label": cat,
                "category": cat,
                "region_type": "furniture",
                "centroid_3d": [round(float(c), 3) for c in c_w],
                "bbox": [int(xs_s.min()), int(ys_s.min()), int(xs_s.max()), int(ys_s.max())],
                "centroid_2d": [round(float(np.mean(xs_s)), 1), round(float(np.mean(ys_s)), 1)],
                "pixel_area": area_scaled,
                "mask": mask_resized,
            })
    return furniture_masks


def match_semantic_regions(
    regions0: list[dict[str, Any]],
    regions1: list[dict[str, Any]],
    max_centroid_dist: float = 0.40,
) -> tuple[list[tuple[int, int, str]], list[int], list[int]]:
    """
    Finds 1-to-1 ground truth correspondences between semantic region masks.
    - Structural planes matched by exact global entity_id
    - Furniture objects matched by 3D world centroid distance & category
    """
    matches = []
    matched_0 = set()
    matched_1 = set()

    # 1. Match Structural Planes
    for i, r0 in enumerate(regions0):
        if r0.get("region_type") == "plane":
            for j, r1 in enumerate(regions1):
                if r1.get("region_type") == "plane" and j not in matched_1:
                    if r0.get("entity_id") == r1.get("entity_id"):
                        matches.append((i, j, r0["label"]))
                        matched_0.add(i)
                        matched_1.add(j)
                        break

    # 2. Match Furniture via 3D World Centroids
    furn_indices_0 = [i for i, r in enumerate(regions0) if r.get("region_type") == "furniture" and i not in matched_0]
    furn_indices_1 = [j for j, r in enumerate(regions1) if r.get("region_type") == "furniture" and j not in matched_1]

    candidates = []
    for i in furn_indices_0:
        c0 = np.array(regions0[i]["centroid_3d"], dtype=np.float32)
        cat0 = regions0[i]["category"]
        for j in furn_indices_1:
            if regions1[j]["category"] == cat0:
                c1 = np.array(regions1[j]["centroid_3d"], dtype=np.float32)
                dist = float(np.linalg.norm(c0 - c1))
                if dist <= max_centroid_dist:
                    candidates.append((dist, i, j, cat0))

    candidates.sort(key=lambda x: x[0])
    for dist, i, j, cat in candidates:
        if i not in matched_0 and j not in matched_1:
            matches.append((i, j, cat))
            matched_0.add(i)
            matched_1.add(j)

    unmatched_0 = [i for i in range(len(regions0)) if i not in matched_0]
    unmatched_1 = [j for j in range(len(regions1)) if j not in matched_1]

    return matches, unmatched_0, unmatched_1


def render_correspondence_vis(
    img0: np.ndarray,
    img1: np.ndarray,
    regions0: list[dict[str, Any]],
    regions1: list[dict[str, Any]],
    matches: list[tuple[int, int, str]],
    unmatched_0: list[int],
    unmatched_1: list[int],
    out_path: Path,
    title: str = "",
) -> None:
    """Renders a side-by-side visual correspondence verification image."""
    H, W = img0.shape[:2]
    canvas = np.zeros((H + 60, W * 2 + 30, 3), dtype=np.uint8)
    canvas.fill(24)

    # Place images
    canvas[50:50+H, :W] = img0
    canvas[50:50+H, W+30:] = img1

    overlay0 = canvas[50:50+H, :W].copy()
    overlay1 = canvas[50:50+H, W+30:].copy()

    # Draw matching regions
    for match_idx, (i, j, label) in enumerate(matches):
        col = DISTINCT_COLORS[match_idx % len(DISTINCT_COLORS)]
        m0 = regions0[i]["mask"] > 0
        m1 = regions1[j]["mask"] > 0

        overlay0[m0] = (overlay0[m0] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
        overlay1[m1] = (overlay1[m1] * 0.45 + np.array(col) * 0.55).astype(np.uint8)

        # Draw contour
        cnts0, _ = cv2.findContours((m0 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts1, _ = cv2.findContours((m1 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay0, cnts0, -1, col, 2)
        cv2.drawContours(overlay1, cnts1, -1, col, 2)

    canvas[50:50+H, :W] = overlay0
    canvas[50:50+H, W+30:] = overlay1

    # Draw connection lines between matching centroids
    for match_idx, (i, j, label) in enumerate(matches):
        col = DISTINCT_COLORS[match_idx % len(DISTINCT_COLORS)]
        c0 = regions0[i]["centroid_2d"]
        c1 = regions1[j]["centroid_2d"]

        pt0 = (int(c0[0]), int(c0[1] + 50))
        pt1 = (int(c1[0] + W + 30), int(c1[1] + 50))

        cv2.circle(canvas, pt0, 5, col, -1)
        cv2.circle(canvas, pt1, 5, col, -1)
        cv2.line(canvas, pt0, pt1, col, 2, cv2.LINE_AA)

        # Label tag
        cv2.putText(canvas, label, (pt0[0] + 5, pt0[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Header title
    cv2.putText(canvas, f"{title} | Matched Regions: {len(matches)} | Unmatched: View0={len(unmatched_0)}, View1={len(unmatched_1)}",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def process_scene_room_pairs(
    scene_dir_str: str,
    output_dir_str: str,
    target_h: int = 480,
    target_w: int = 640,
    min_matches: int = 3,
    max_rotation_deg: float = 85.0,
) -> dict[str, Any]:
    """Extracts semantic region mask pairs for all viewpoint combinations in a scene."""
    scene_dir = Path(scene_dir_str)
    output_dir = Path(output_dir_str)
    scene_id = scene_dir.name

    room_dirs = sorted([d for d in (scene_dir / "2D_rendering").iterdir() if d.is_dir()])
    pairs_saved = []

    for room in room_dirs:
        persp_dir = room / "perspective" / "full"
        if not persp_dir.is_dir():
            continue

        view_dirs = sorted([v for v in persp_dir.iterdir() if v.is_dir() and (v / "camera_pose.txt").is_file()])
        if len(view_dirs) < 2:
            continue

        # Extract views
        views_dict = {}
        for vd in view_dirs:
            try:
                view_obj = Structured3DView(scene_id, room.name, vd.name, vd)
                bgr = view_obj.load_rgb()
                bgr_resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                planes = extract_plane_masks(view_obj, target_h, target_w)
                furns = extract_furniture_masks(view_obj, target_h, target_w)
                all_regions = planes + furns

                if len(all_regions) >= 2:
                    views_dict[vd.name] = {
                        "view_id": vd.name,
                        "bgr": bgr_resized,
                        "regions": all_regions,
                        "cam": view_obj.camera,
                    }
            except Exception:
                continue

        # Form pairs
        v_keys = list(views_dict.keys())
        for a_i in range(len(v_keys)):
            for b_i in range(a_i + 1, len(v_keys)):
                va = views_dict[v_keys[a_i]]
                vb = views_dict[v_keys[b_i]]

                cam_a, cam_b = va["cam"], vb["cam"]
                rot_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(cam_a.R_c2w.T @ cam_b.R_c2w) - 1.0) / 2.0, -1.0, 1.0))))
                base_m = float(np.linalg.norm(cam_a.position - cam_b.position) / 1000.0)

                if rot_deg > max_rotation_deg or (rot_deg < 10.0 and base_m < 0.3):
                    continue

                matches, un0, un1 = match_semantic_regions(va["regions"], vb["regions"])
                if len(matches) < min_matches:
                    continue

                # Save Pair Folder
                pair_id = f"{scene_id}_{room.name}_v{va['view_id']}_v{vb['view_id']}"
                pair_dir = output_dir / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)

                # Save RGB images
                cv2.imwrite(str(pair_dir / "image_0.png"), va["bgr"])
                cv2.imwrite(str(pair_dir / "image_1.png"), vb["bgr"])

                # Save Masks for View 0
                masks0_dir = pair_dir / "masks_0"
                masks0_dir.mkdir(exist_ok=True)
                meta_regions_0 = []
                for idx, r in enumerate(va["regions"]):
                    cv2.imwrite(str(masks0_dir / f"mask_{idx:02d}.png"), r["mask"])
                    meta_regions_0.append({
                        "idx": idx,
                        "label": r["label"],
                        "category": r["category"],
                        "region_type": r["region_type"],
                        "bbox": r["bbox"],
                        "centroid_2d": r["centroid_2d"],
                        "pixel_area": r["pixel_area"],
                    })

                # Save Masks for View 1
                masks1_dir = pair_dir / "masks_1"
                masks1_dir.mkdir(exist_ok=True)
                meta_regions_1 = []
                for idx, r in enumerate(vb["regions"]):
                    cv2.imwrite(str(masks1_dir / f"mask_{idx:02d}.png"), r["mask"])
                    meta_regions_1.append({
                        "idx": idx,
                        "label": r["label"],
                        "category": r["category"],
                        "region_type": r["region_type"],
                        "bbox": r["bbox"],
                        "centroid_2d": r["centroid_2d"],
                        "pixel_area": r["pixel_area"],
                    })

                with open(pair_dir / "regions_0.json", "w", encoding="utf-8") as f:
                    json.dump(meta_regions_0, f, indent=2)
                with open(pair_dir / "regions_1.json", "w", encoding="utf-8") as f:
                    json.dump(meta_regions_1, f, indent=2)

                # Save Correspondence JSON
                match_records = [{"mask0_idx": m[0], "mask1_idx": m[1], "label": m[2]} for m in matches]
                corr_dict = {
                    "pair_id": pair_id,
                    "scene_id": scene_id,
                    "room_id": room.name,
                    "view0_id": va["view_id"],
                    "view1_id": vb["view_id"],
                    "rotation_deg": round(rot_deg, 2),
                    "baseline_m": round(base_m, 2),
                    "num_matches": len(matches),
                    "matches": match_records,
                    "unmatched_in_view0": un0,
                    "unmatched_in_view1": un1,
                }
                with open(pair_dir / "correspondence.json", "w", encoding="utf-8") as f:
                    json.dump(corr_dict, f, indent=2)

                # Export visualization
                render_correspondence_vis(
                    va["bgr"], vb["bgr"], va["regions"], vb["regions"],
                    matches, un0, un1, pair_dir / "vis_correspondence.jpg",
                    title=f"{scene_id} ({room.name}) View {va['view_id']} <-> View {vb['view_id']}"
                )

                pairs_saved.append(pair_id)

    return {"scene_id": scene_id, "pairs_created": len(pairs_saved), "pair_ids": pairs_saved}


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured3D Semantic Region Mask Matching Dataset Builder")
    parser.add_argument("--s3d-dir", type=str, default="Structured3D_perspective_full_00/Structured3D", help="Structured3D directory")
    parser.add_argument("--out-dir", type=str, default="data/structured3d_semantic_regions", help="Output dataset directory")
    parser.add_argument("--max-scenes", type=int, default=50, help="Max scenes to process")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 16, help="Worker processes")
    parser.add_argument("--min-matches", type=int, default=3, help="Minimum matching regions per pair")
    args = parser.parse_args()

    s3d_dir = Path(args.s3d_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted([d for d in s3d_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")])
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    print("=" * 75)
    print("      Structured3D Semantic Region Mask Matching Dataset Builder      ")
    print("=" * 75)
    print(f"  Source Directory: {s3d_dir}")
    print(f"  Output Directory: {out_dir}")
    print(f"  Target Scenes:    {len(scenes)}")
    print(f"  Worker Processes: {args.num_workers}")
    print("=" * 75 + "\n")

    t0 = time.time()
    all_pair_ids = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_scene_room_pairs, str(s), str(out_dir),
                min_matches=args.min_matches
            ): s.name for s in scenes
        }

        for idx, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res["pairs_created"] > 0:
                all_pair_ids.extend(res["pair_ids"])
            print(f"  [{idx+1:3d}/{len(scenes):3d}] Scene {res['scene_id']} -> {res['pairs_created']} region mask pairs extracted.")

    all_pair_ids.sort()
    n_pairs = len(all_pair_ids)
    n_train = int(n_pairs * 0.8)
    n_val = int(n_pairs * 0.1)

    train_pairs = all_pair_ids[:n_train]
    val_pairs = all_pair_ids[n_train:n_train+n_val]
    test_pairs = all_pair_ids[n_train+n_val:]

    with open(out_dir / "train_split.json", "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2)
    with open(out_dir / "val_split.json", "w", encoding="utf-8") as f:
        json.dump(val_pairs, f, indent=2)
    with open(out_dir / "test_split.json", "w", encoding="utf-8") as f:
        json.dump(test_pairs, f, indent=2)

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print("  [✓] SEMANTIC REGION MASK DATASET GENERATION COMPLETE!")
    print(f"  • Total Valid Pairs: {n_pairs} (Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)})")
    print(f"  • Time Elapsed:      {elapsed:.1f}s ({n_pairs / max(elapsed, 1e-3):.1f} pairs/sec)")
    print(f"  • Splits Exported:   train_split.json, val_split.json, test_split.json")
    print("=" * 75)


if __name__ == "__main__":
    main()
