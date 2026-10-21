#!/usr/bin/env python3
"""
Batch Ground-Truth 2D Line Matching Dataset Builder.

Processes the entire Structured3D dataset using multi-processing:
  - Computes ground-truth line matches for all valid viewpoint pairs in the same room.
  - Applies near-plane clipping, frustum boundaries, and depth-buffer occlusion culling.
  - Filters out pairs with excessive rotation or insufficient baseline / matching lines.
  - Exports per-scene annotations and a unified global index for instant training loading.

Usage:
  python tools/build_line_gt_dataset.py
  python tools/build_line_gt_dataset.py --num-workers 16
  python tools/build_line_gt_dataset.py --min-matches 3 --max-rotation 75
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
from PIL import Image

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.read_structured3d import (
    STRUCTURED3D_CLASSES,
    Camera,
    Structured3DDataset,
    Structured3DScene,
    Structured3DView,
)
from datasets.ground_truth import check_line_visibility_and_project, query_line_semantic


def process_single_scene(
    scene_dir_str: str,
    output_scene_dir_str: str,
    min_length: float = 20.0,
    min_vis_ratio: float = 0.65,
    occlusion_tolerance_mm: float = 120.0,
    min_matches: int = 1,
    max_rotation_deg: float = 85.0,
    min_baseline_m: float = 0.15,
    max_baseline_m: float = 4.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Worker function to process all viewpoint pairs within a single scene.
    """
    scene_dir = Path(scene_dir_str)
    scene_id = scene_dir.name
    out_json = Path(output_scene_dir_str) / f"{scene_id}.json"

    if out_json.is_file() and not overwrite:
        with open(out_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {
            "scene_id": scene_id,
            "pairs_count": len(existing.get("pairs", [])),
            "skipped": True,
        }

    anno_path = scene_dir / "annotation_3d.json"
    if not anno_path.is_file():
        return {"scene_id": scene_id, "pairs_count": 0, "skipped": False}

    with open(anno_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    junctions = {
        j["ID"]: np.array(j["coordinate"], dtype=np.float64)
        for j in anno.get("junctions", [])
        if "ID" in j and "coordinate" in j
    }
    matrix = np.array(anno.get("lineJunctionMatrix", []))
    lines_meta = anno.get("lines", [])

    # Collect 3D lines
    lines_3d: list[tuple[int, np.ndarray, np.ndarray]] = []
    if matrix.ndim == 2:
        for idx, row in enumerate(matrix):
            j_idx = np.where(row == 1)[0]
            if len(j_idx) == 2 and j_idx[0] in junctions and j_idx[1] in junctions:
                lid = lines_meta[idx]["ID"] if idx < len(lines_meta) else idx
                lines_3d.append((lid, junctions[j_idx[0]], junctions[j_idx[1]]))

    # Scan viewpoints in this scene
    pattern = scene_dir / "2D_rendering" / "*" / "perspective" / "full" / "*"
    views_by_room: dict[str, list[Path]] = {}
    for vp in sorted(glob.glob(str(pattern))):
        p = Path(vp)
        if p.is_dir() and (p / "camera_pose.txt").is_file():
            room_id = p.parts[-4]
            views_by_room.setdefault(room_id, []).append(p)

    scene_pairs_data: list[dict[str, Any]] = []

    # Process pairs per room
    for room_id, vpaths in views_by_room.items():
        n = len(vpaths)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                p0, p1 = vpaths[i], vpaths[j]
                v0_id, v1_id = p0.parts[-1], p1.parts[-1]

                cam0 = Camera.from_file(p0 / "camera_pose.txt")
                cam1 = Camera.from_file(p1 / "camera_pose.txt")

                R_rel, t_rel, rot_deg, base_m = cam0.relative_pose_to(cam1)
                if not (min_baseline_m <= base_m <= max_baseline_m):
                    continue
                if rot_deg > max_rotation_deg:
                    continue

                # Load depth maps
                depth0 = cv2.imread(str(p0 / "depth.png"), cv2.IMREAD_ANYDEPTH)
                depth1 = cv2.imread(str(p1 / "depth.png"), cv2.IMREAD_ANYDEPTH)
                if depth0 is None or depth1 is None:
                    continue

                sem0 = np.array(Image.open(p0 / "semantic.png"), dtype=np.uint8)

                # Match lines
                pair_matches: list[dict[str, Any]] = []
                for lid, p1_w, p2_w in lines_3d:
                    l0, r0 = check_line_visibility_and_project(
                        cam0,
                        depth0,
                        p1_w,
                        p2_w,
                        min_length=min_length,
                        min_vis_ratio=min_vis_ratio,
                        occlusion_tolerance_mm=occlusion_tolerance_mm,
                    )
                    if l0 is None:
                        continue
                    l1, r1 = check_line_visibility_and_project(
                        cam1,
                        depth1,
                        p1_w,
                        p2_w,
                        min_length=min_length,
                        min_vis_ratio=min_vis_ratio,
                        occlusion_tolerance_mm=occlusion_tolerance_mm,
                    )
                    if l1 is None:
                        continue

                    sem_name = query_line_semantic(sem0, l0)
                    len0 = math.hypot(l0[1][0] - l0[0][0], l0[1][1] - l0[0][1])
                    len1 = math.hypot(l1[1][0] - l1[0][0], l1[1][1] - l1[0][1])

                    pair_matches.append(
                        {
                            "line_id": lid,
                            "semantic": sem_name,
                            "line0": {"start": l0[0], "end": l0[1], "length": round(len0, 1)},
                            "line1": {"start": l1[0], "end": l1[1], "length": round(len1, 1)},
                            "visibility0": round(r0, 3),
                            "visibility1": round(r1, 3),
                        }
                    )

                if len(pair_matches) >= min_matches:
                    scene_pairs_data.append(
                        {
                            "pair_id": f"{scene_id}_{room_id}_{v0_id}_{v1_id}",
                            "scene_id": scene_id,
                            "room_id": room_id,
                            "view0_id": v0_id,
                            "view1_id": v1_id,
                            "view0_dir": str(p0.relative_to(ROOT)),
                            "view1_dir": str(p1.relative_to(ROOT)),
                            "relative_rotation_deg": round(rot_deg, 2),
                            "baseline_meters": round(base_m, 3),
                            "num_matches": len(pair_matches),
                            "matches": pair_matches,
                        }
                    )

    # Save scene json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene_id": scene_id,
                "total_valid_pairs": len(scene_pairs_data),
                "pairs": scene_pairs_data,
            },
            f,
            indent=2,
        )

    return {
        "scene_id": scene_id,
        "pairs_count": len(scene_pairs_data),
        "skipped": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Ground-Truth 2D Line Matching Dataset Builder"
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        type=Path,
        default=None,
        help="Structured3D dataset root directory (default: auto-detected)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "line_gt_dataset",
        help="Directory to store precomputed line ground truth dataset (default: data/line_gt_dataset)",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel worker processes (default: 16)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=1,
        help="Minimum matching lines required to keep a pair (default: 1)",
    )
    parser.add_argument(
        "--min-line-length",
        type=float,
        default=20.0,
        help="Minimum 2D line length in pixels (default: 20.0)",
    )
    parser.add_argument(
        "--max-rotation",
        type=float,
        default=85.0,
        help="Maximum relative camera rotation angle in degrees (default: 85.0)",
    )
    parser.add_argument(
        "--min-baseline",
        type=float,
        default=0.15,
        help="Minimum camera baseline in meters (default: 0.15)",
    )
    parser.add_argument(
        "--max-baseline",
        type=float,
        default=4.5,
        help="Maximum camera baseline in meters (default: 4.5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene annotations if already generated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("Building Structured3D Ground-Truth Line Matching Dataset")
    print("=" * 70)

    dataset = Structured3DDataset(args.dataset_dir)
    scenes_dirs = sorted([s.scene_dir for s in dataset.scenes.values()])
    print(f"Total scenes to process: {len(scenes_dirs)}")
    print(f"Output directory:        {args.output_dir}")
    print(f"Workers:                 {args.num_workers}")
    print(f"Filtering criteria:      min_matches={args.min_matches}, max_rot={args.max_rotation}°, baseline=[{args.min_baseline}m, {args.max_baseline}m]")
    print("-" * 70)

    scenes_out_dir = args.output_dir / "scenes"
    scenes_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_valid_pairs = 0
    total_matches_count = 0
    processed_count = 0
    global_index_pairs: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_scene,
                str(sdir),
                str(scenes_out_dir),
                args.min_line_length,
                0.65,
                120.0,
                args.min_matches,
                args.max_rotation,
                args.min_baseline,
                args.max_baseline,
                args.overwrite,
            ): sdir.name
            for sdir in scenes_dirs
        }

        for future in as_completed(futures):
            res = future.result()
            processed_count += 1
            scene_id = res["scene_id"]
            p_cnt = res["pairs_count"]
            total_valid_pairs += p_cnt

            elapsed = max(time.time() - t0, 1e-4)
            speed = processed_count / elapsed
            eta = (len(scenes_dirs) - processed_count) / max(speed, 1e-4)

            print(
                f"\r[{processed_count:3d}/{len(scenes_dirs):3d}] Scene: {scene_id} "
                f"-> {p_cnt:2d} pairs | Total pairs: {total_valid_pairs:4d} | "
                f"{speed:.1f} scenes/s | ETA: {eta:.1f}s",
                end="",
                flush=True,
            )

    print()
    print("-" * 70)
    print("Building global index...")

    # Load all scene jsons to build unified global index
    for scene_json_path in sorted(scenes_out_dir.glob("*.json")):
        with open(scene_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("pairs", []):
            total_matches_count += p["num_matches"]
            # Save lightweight entry to global index
            global_index_pairs.append(
                {
                    "pair_id": p["pair_id"],
                    "scene_id": p["scene_id"],
                    "room_id": p["room_id"],
                    "view0_id": p["view0_id"],
                    "view1_id": p["view1_id"],
                    "view0_dir": p["view0_dir"],
                    "view1_dir": p["view1_dir"],
                    "relative_rotation_deg": p["relative_rotation_deg"],
                    "baseline_meters": p["baseline_meters"],
                    "num_matches": p["num_matches"],
                    "scene_annotation": f"scenes/{scene_json_path.name}",
                }
            )

    index_out_path = args.output_dir / "index.json"
    summary_data = {
        "dataset_root": str(dataset.root_dir),
        "total_scenes": len(scenes_dirs),
        "total_valid_pairs": len(global_index_pairs),
        "total_line_matches": total_matches_count,
        "avg_matches_per_pair": round(total_matches_count / max(len(global_index_pairs), 1), 2),
        "filtering": {
            "min_matches": args.min_matches,
            "min_line_length": args.min_line_length,
            "max_rotation_deg": args.max_rotation,
            "min_baseline_m": args.min_baseline,
            "max_baseline_m": args.max_baseline,
        },
        "pairs": global_index_pairs,
    }

    with open(index_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    total_time = time.time() - t0
    print("=" * 70)
    print(f"Dataset Build Finished in {total_time:.2f} seconds!")
    print(f"  Processed scenes:           {len(scenes_dirs)}")
    print(f"  Total valid matching pairs: {len(global_index_pairs)}")
    print(f"  Total GT line matches:      {total_matches_count}")
    print(f"  Avg matches / pair:         {summary_data['avg_matches_per_pair']}")
    print(f"  Global Index Path:          {index_out_path}")
    print(f"  Per-Scene Annotations:      {scenes_out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
