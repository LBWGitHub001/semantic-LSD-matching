"""
LineFeat: Ground-Truth 2D Line Matching Generator (3D Wireframe Projection).

Implements Option 1:
  - Projects 3D wireframe structural lines from Structured3D annotation_3d.json
  - Frustum clipping & near-plane clipping
  - Z-buffer occlusion culling against measured depth maps
  - Semantic label querying for line segments
  - High-performance multi-processing batch calculation for the entire dataset
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.read_structured3d import (
    STRUCTURED3D_CLASSES,
    Camera,
    Structured3DDataset,
    Structured3DScene,
    Structured3DView,
)


@dataclass
class GTLineMatch:
    """A pair of 2D line segments that correspond to the exact same 3D physical line."""

    line_id: int  # Global 3D line ID from annotation_3d.json
    line0: tuple[tuple[int, int], tuple[int, int]]  # ((u1, v1), (u2, v2)) in view 0
    line1: tuple[tuple[int, int], tuple[int, int]]  # ((u1, v1), (u2, v2)) in view 1
    length0: float  # Length in pixels
    length1: float
    vis_ratio0: float  # Unoccluded ratio in view 0 (0.0 to 1.0)
    vis_ratio1: float  # Unoccluded ratio in view 1 (0.0 to 1.0)
    semantic_name: str  # Semantic class name (wall, door, ceiling, etc.)


def get_distinct_colors(num_colors: int) -> list[tuple[int, int, int]]:
    """Generate visually distinct BGR colors for line matching visualization."""
    colors: list[tuple[int, int, int]] = []
    for i in range(num_colors):
        hue = int(180 * (i / max(num_colors, 1)))
        hsv_pixel = np.uint8([[[hue, 230, 240]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def check_line_visibility_and_project(
    camera: Camera,
    depth_map: np.ndarray,
    p1_world: np.ndarray,
    p2_world: np.ndarray,
    min_length: float = 15.0,
    num_samples: int = 25,
    occlusion_tolerance_mm: float = 120.0,
    min_vis_ratio: float = 0.65,
) -> tuple[tuple[tuple[int, int], tuple[int, int]] | None, float]:
    """
    Project a 3D line segment to 2D image and verify occlusion against depth map.

    Returns:
        line_2d: ((u1, v1), (u2, v2)) or None
        visibility_ratio: float in [0.0, 1.0]
    """
    proj = camera.project_line_segment(p1_world, p2_world)
    if proj is None:
        return None, 0.0

    (u1, v1), (u2, v2) = proj
    length = math.hypot(u2 - u1, v2 - v1)
    if length < min_length:
        return None, 0.0

    # Camera 3D depths
    p1_cam = camera.world_to_cam(p1_world)
    p2_cam = camera.world_to_cam(p2_world)
    z1 = max(float(p1_cam[2]), 1.0)
    z2 = max(float(p2_cam[2]), 1.0)

    # Sample along 2D line segment
    ts = np.linspace(0.0, 1.0, num_samples)
    us = np.clip(np.round((1.0 - ts) * u1 + ts * u2).astype(int), 0, camera.width - 1)
    vs = np.clip(np.round((1.0 - ts) * v1 + ts * v2).astype(int), 0, camera.height - 1)

    # True projected depth along ray: 1/Z is linear in screen space
    inv_z = (1.0 - ts) / z1 + ts / z2
    z_proj = 1.0 / np.maximum(inv_z, 1e-6)

    # Measured depth from depth map (in mm)
    z_meas = depth_map[vs, us].astype(np.float64)

    # Unoccluded check: depth must be positive and not blocked by foreground
    unoccluded = (z_meas > 0) & (z_proj <= z_meas + occlusion_tolerance_mm)
    vis_ratio = float(np.mean(unoccluded))

    if vis_ratio < min_vis_ratio:
        return None, vis_ratio

    return proj, vis_ratio


def query_line_semantic(
    semantic_map: np.ndarray,
    line_2d: tuple[tuple[int, int], tuple[int, int]],
    num_samples: int = 20,
) -> str:
    """Sample pixels along the 2D line in the semantic map to get dominant class name."""
    (u1, v1), (u2, v2) = line_2d
    h, w = semantic_map.shape[:2]
    ts = np.linspace(0.0, 1.0, num_samples)
    us = np.clip(np.round((1.0 - ts) * u1 + ts * u2).astype(int), 0, w - 1)
    vs = np.clip(np.round((1.0 - ts) * v1 + ts * v2).astype(int), 0, h - 1)

    classes = semantic_map[vs, us]
    valid = classes[classes > 0]
    if len(valid) == 0:
        return "void"
    vals, counts = np.unique(valid, return_counts=True)
    dom_id = vals[np.argmax(counts)]
    if dom_id < len(STRUCTURED3D_CLASSES):
        return STRUCTURED3D_CLASSES[dom_id]
    return f"class_{dom_id}"


def extract_ground_truth_line_matches(
    view0: Structured3DView,
    view1: Structured3DView,
    scene: Structured3DScene,
    min_length: float = 20.0,
    min_vis_ratio: float = 0.65,
    occlusion_tolerance_mm: float = 120.0,
) -> list[GTLineMatch]:
    """
    Extract all 100% verified ground-truth matching line segments between view0 and view1.
    """
    depth0 = view0.load_depth(in_meters=False)
    depth1 = view1.load_depth(in_meters=False)
    sem0 = view0.load_semantic()

    anno = scene.load_annotation_3d()
    junctions = {
        j["ID"]: np.array(j["coordinate"], dtype=np.float64)
        for j in anno.get("junctions", [])
        if "ID" in j and "coordinate" in j
    }
    matrix = np.array(anno.get("lineJunctionMatrix", []))
    lines_meta = anno.get("lines", [])

    cam0 = view0.camera
    cam1 = view1.camera

    matches: list[GTLineMatch] = []

    for idx, row in enumerate(matrix):
        j_idx = np.where(row == 1)[0]
        if len(j_idx) != 2:
            continue
        id1, id2 = j_idx[0], j_idx[1]
        if id1 not in junctions or id2 not in junctions:
            continue

        p1_w = junctions[id1]
        p2_w = junctions[id2]
        line_id = lines_meta[idx]["ID"] if idx < len(lines_meta) else idx

        # Visibility and projection in View 0
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

        # Visibility and projection in View 1
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

        len0 = math.hypot(l0[1][0] - l0[0][0], l0[1][1] - l0[0][1])
        len1 = math.hypot(l1[1][0] - l1[0][0], l1[1][1] - l1[0][1])
        sem_name = query_line_semantic(sem0, l0)

        matches.append(
            GTLineMatch(
                line_id=line_id,
                line0=l0,
                line1=l1,
                length0=len0,
                length1=len1,
                vis_ratio0=r0,
                vis_ratio1=r1,
                semantic_name=sem_name,
            )
        )

    matches.sort(key=lambda m: m.length0 + m.length1, reverse=True)
    return matches


def draw_ground_truth_matches(
    view0: Structured3DView,
    view1: Structured3DView,
    matches: list[GTLineMatch],
    max_draw: int = 30,
    show_connecting_lines: bool = True,
) -> np.ndarray:
    """
    Render side-by-side verification image.
    Matched lines share the exact same distinctive color and label ID across both views.
    """
    img0 = view0.load_rgb(rawlight=True)
    img1 = view1.load_rgb(rawlight=True)
    h, w = img0.shape[:2]

    canvas = np.hstack([img0.copy(), img1.copy()])
    draw_matches = matches[:max_draw]
    colors = get_distinct_colors(len(draw_matches))

    midpoints0: list[tuple[int, int]] = []
    midpoints1: list[tuple[int, int]] = []

    for i, m in enumerate(draw_matches):
        col = colors[i]
        # Line in view 0
        p1_0, p2_0 = m.line0
        cv2.line(canvas, p1_0, p2_0, col, 3, cv2.LINE_AA)
        cv2.circle(canvas, p1_0, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2_0, 4, (255, 255, 255), -1, cv2.LINE_AA)

        mid0 = (int((p1_0[0] + p2_0[0]) / 2), int((p1_0[1] + p2_0[1]) / 2))
        midpoints0.append(mid0)

        tag0 = f"L{m.line_id}:{m.semantic_name}"
        cv2.putText(canvas, tag0, (mid0[0] + 5, mid0[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, tag0, (mid0[0] + 5, mid0[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Line in view 1 (shifted by w)
        p1_1 = (m.line1[0][0] + w, m.line1[0][1])
        p2_1 = (m.line1[1][0] + w, m.line1[1][1])
        cv2.line(canvas, p1_1, p2_1, col, 3, cv2.LINE_AA)
        cv2.circle(canvas, p1_1, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2_1, 4, (255, 255, 255), -1, cv2.LINE_AA)

        mid1 = (int((p1_1[0] + p2_1[0]) / 2), int((p1_1[1] + p2_1[1]) / 2))
        midpoints1.append(mid1)

        tag1 = f"L{m.line_id}"
        cv2.putText(canvas, tag1, (mid1[0] + 5, mid1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, tag1, (mid1[0] + 5, mid1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if show_connecting_lines:
        overlay = canvas.copy()
        for i in range(len(midpoints0)):
            cv2.line(overlay, midpoints0[i], midpoints1[i], colors[i], 1, cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)

    R_rel, t_rel, rot_deg, base_m = view0.camera.relative_pose_to(view1.camera)
    info_text = (
        f"Ground Truth Line Matches: {len(matches)} pairs | "
        f"{view0.scene_id} room_{view0.room_id} (View {view0.view_id} <-> View {view1.view_id}) | "
        f"Rotation: {rot_deg:.1f} deg | Baseline: {base_m:.2f} m"
    )

    header_bar = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header_bar, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header_bar, canvas])


def process_scene_lines(
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
    """Process all valid viewpoint pairs in a single scene and export JSON."""
    scene_dir = Path(scene_dir_str)
    scene_id = scene_dir.name
    out_json = Path(output_scene_dir_str) / f"{scene_id}.json"

    if out_json.is_file() and not overwrite:
        with open(out_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {"scene_id": scene_id, "pairs_count": len(existing.get("pairs", [])), "skipped": True}

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

    lines_3d: list[tuple[int, np.ndarray, np.ndarray]] = []
    if matrix.ndim == 2:
        for idx, row in enumerate(matrix):
            j_idx = np.where(row == 1)[0]
            if len(j_idx) == 2 and j_idx[0] in junctions and j_idx[1] in junctions:
                lid = lines_meta[idx]["ID"] if idx < len(lines_meta) else idx
                lines_3d.append((lid, junctions[j_idx[0]], junctions[j_idx[1]]))

    pattern = scene_dir / "2D_rendering" / "*" / "perspective" / "full" / "*"
    views_by_room: dict[str, list[Path]] = {}
    for vp in sorted(glob.glob(str(pattern))):
        p = Path(vp)
        if p.is_dir() and (p / "camera_pose.txt").is_file():
            room_id = p.parts[-4]
            views_by_room.setdefault(room_id, []).append(p)

    scene_pairs_data: list[dict[str, Any]] = []

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

                depth0 = cv2.imread(str(p0 / "depth.png"), cv2.IMREAD_ANYDEPTH)
                depth1 = cv2.imread(str(p1 / "depth.png"), cv2.IMREAD_ANYDEPTH)
                if depth0 is None or depth1 is None:
                    continue

                sem0 = np.array(Image.open(p0 / "semantic.png"), dtype=np.uint8)

                pair_matches: list[dict[str, Any]] = []
                for lid, p1_w, p2_w in lines_3d:
                    l0, r0 = check_line_visibility_and_project(
                        cam0, depth0, p1_w, p2_w, min_length, min_vis_ratio, occlusion_tolerance_mm
                    )
                    if l0 is None:
                        continue
                    l1, r1 = check_line_visibility_and_project(
                        cam1, depth1, p1_w, p2_w, min_length, min_vis_ratio, occlusion_tolerance_mm
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

    return {"scene_id": scene_id, "pairs_count": len(scene_pairs_data), "skipped": False}


def build_line_gt_dataset(
    output_dir: str | Path,
    dataset_dir: str | Path | None = None,
    num_workers: int = 16,
    min_matches: int = 1,
    min_line_length: float = 20.0,
    max_rotation_deg: float = 85.0,
    min_baseline_m: float = 0.15,
    max_baseline_m: float = 4.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Multi-process batch processor across all scenes."""
    out_dir = Path(output_dir)
    dataset = Structured3DDataset(dataset_dir)
    scenes_dirs = sorted([s.scene_dir for s in dataset.scenes.values()])

    scenes_out_dir = out_dir / "scenes"
    scenes_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_valid_pairs = 0
    total_matches_count = 0
    global_index_pairs: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_scene_lines,
                str(sdir),
                str(scenes_out_dir),
                min_line_length,
                0.65,
                120.0,
                min_matches,
                max_rotation_deg,
                min_baseline_m,
                max_baseline_m,
                overwrite,
            ): sdir.name
            for sdir in scenes_dirs
        }
        for future in as_completed(futures):
            res = future.result()
            total_valid_pairs += res["pairs_count"]

    for scene_json_path in sorted(scenes_out_dir.glob("*.json")):
        with open(scene_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for p in data.get("pairs", []):
            total_matches_count += p["num_matches"]
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

    index_out_path = out_dir / "index.json"
    summary_data = {
        "dataset_root": str(dataset.root_dir),
        "total_scenes": len(scenes_dirs),
        "total_valid_pairs": len(global_index_pairs),
        "total_line_matches": total_matches_count,
        "avg_matches_per_pair": round(total_matches_count / max(len(global_index_pairs), 1), 2),
        "filtering": {
            "min_matches": min_matches,
            "min_line_length": min_line_length,
            "max_rotation_deg": max_rotation_deg,
            "min_baseline_m": min_baseline_m,
            "max_baseline_m": max_baseline_m,
        },
        "pairs": global_index_pairs,
    }

    with open(index_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data
