#!/usr/bin/env python3
"""
Ground Truth 2D Line Matching Generator (Option 1: 3D Wireframe Projection).

Projects 3D structural wireframe lines from Structured3D annotation_3d.json
into pairs of camera viewpoints, applies near-plane clipping, image border clipping,
and depth-buffer occlusion culling, producing 100% verified ground-truth line correspondences.

Usage:
  python tools/generate_line_gt.py
  python tools/generate_line_gt.py --scene scene_00010 --room 843531 --view0 0 --view1 3
  python tools/generate_line_gt.py --min-matches 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

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
    semantic_name: str  # Semantic class name (wall, door, etc.)


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
    # Filter out void (0) if other classes exist
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
    Extract all 100% ground-truth matching line segments between view0 and view1.
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

    # Sort matches by length descending
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

    # Side by side canvas
    canvas = np.hstack([img0.copy(), img1.copy()])
    draw_matches = matches[:max_draw]
    colors = get_distinct_colors(len(draw_matches))

    # Midpoints for connecting lines
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

        # Label in view 0
        tag0 = f"L{m.line_id}:{m.semantic_name}"
        cv2.putText(
            canvas,
            tag0,
            (mid0[0] + 5, mid0[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            tag0,
            (mid0[0] + 5, mid0[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Line in view 1 (shift x by w)
        p1_1 = (m.line1[0][0] + w, m.line1[0][1])
        p2_1 = (m.line1[1][0] + w, m.line1[1][1])
        cv2.line(canvas, p1_1, p2_1, col, 3, cv2.LINE_AA)
        cv2.circle(canvas, p1_1, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2_1, 4, (255, 255, 255), -1, cv2.LINE_AA)

        mid1 = (int((p1_1[0] + p2_1[0]) / 2), int((p1_1[1] + p2_1[1]) / 2))
        midpoints1.append(mid1)

        # Label in view 1
        tag1 = f"L{m.line_id}"
        cv2.putText(
            canvas,
            tag1,
            (mid1[0] + 5, mid1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            tag1,
            (mid1[0] + 5, mid1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw thin dashed connecting lines between midpoints
    if show_connecting_lines:
        overlay = canvas.copy()
        for i in range(len(midpoints0)):
            col = colors[i]
            cv2.line(overlay, midpoints0[i], midpoints1[i], col, 1, cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)

    # Header info
    R_rel, t_rel, rot_deg, base_m = view0.camera.relative_pose_to(view1.camera)
    info_text = (
        f"Ground Truth Line Matches: {len(matches)} pairs | "
        f"{view0.scene_id} room_{view0.room_id} (View {view0.view_id} <-> View {view1.view_id}) | "
        f"Rotation: {rot_deg:.1f} deg | Baseline: {base_m:.2f} m"
    )

    header_bar = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header_bar,
        info_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    res = np.vstack([header_bar, canvas])
    return res


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ground Truth Line Matching Generator (3D Wireframe Projection)"
    )
    parser.add_argument(
        "-s", "--scene", type=str, default=None, help="Scene ID (e.g. scene_00010)"
    )
    parser.add_argument("--room", type=str, default=None, help="Room ID")
    parser.add_argument("--view0", type=str, default=None, help="First view ID")
    parser.add_argument("--view1", type=str, default=None, help="Second view ID")
    parser.add_argument(
        "--min-matches",
        type=int,
        default=6,
        help="Minimum required GT line matches when auto-searching (default: 6)",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=25,
        help="Maximum line matches to draw in visual output (default: 25)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "outputs" / "line_gt" / "ground_truth_line_matches.jpg",
        help="Output image path for visualization",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "outputs" / "line_gt" / "ground_truth_line_matches.json",
        help="Output json path for machine-readable ground truth data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading Structured3D dataset...")
    dataset = Structured3DDataset()

    view0: Structured3DView | None = None
    view1: Structured3DView | None = None
    target_scene: Structured3DScene | None = None

    if args.scene:
        if args.scene not in dataset.scenes:
            raise KeyError(f"Scene {args.scene} not found.")
        target_scene = dataset.scenes[args.scene]
        views = target_scene.get_views()
        if args.room:
            views = [v for v in views if v.room_id == args.room]

        if args.view0 and args.view1:
            v0_list = [v for v in views if v.view_id == args.view0]
            v1_list = [v for v in views if v.view_id == args.view1]
            if not v0_list or not v1_list:
                raise ValueError("Specified views not found in room/scene.")
            view0, view1 = v0_list[0], v1_list[0]
        else:
            pairs = target_scene.get_view_pairs(args.room)
            if not pairs:
                raise ValueError("No matching view pairs in specified scene/room.")
            view0, view1 = pairs[0]
    else:
        # Search dataset for an optimal pair with high line overlap
        print("Searching for an optimal scene pair with rich line overlap...")
        # Check scene_00010 room 843531 or scene_00005
        candidate_scenes = ["scene_00010", "scene_00005", "scene_00002", "scene_00000"]
        best_pair = None
        best_matches = []
        best_scene = None

        for sid in candidate_scenes:
            if sid not in dataset.scenes:
                continue
            sc = dataset.scenes[sid]
            pairs = sc.get_view_pairs()
            for p0, p1 in pairs:
                # Filter by mild rotation
                _, _, rot_deg, base_m = p0.camera.relative_pose_to(p1.camera)
                if rot_deg > 55.0 or base_m < 0.2:
                    continue
                m = extract_ground_truth_line_matches(p0, p1, sc)
                if len(m) > len(best_matches):
                    best_matches = m
                    best_pair = (p0, p1)
                    best_scene = sc
                if len(best_matches) >= args.min_matches:
                    break
            if len(best_matches) >= args.min_matches:
                break

        if best_pair is None:
            raise RuntimeError("Could not find suitable candidate pair in dataset.")
        view0, view1 = best_pair
        target_scene = best_scene

    print("=" * 70)
    print("Found Ground Truth Line Matching Pair:")
    print(f"  Scene:     {target_scene.scene_id}")
    print(f"  Room:      {view0.room_id}")
    print(f"  View 0:    {view0.view_id}")
    print(f"  View 1:    {view1.view_id}")

    R_rel, t_rel, rot_deg, base_m = view0.camera.relative_pose_to(view1.camera)
    print(f"  Rotation:  {rot_deg:.2f}°")
    print(f"  Baseline:  {base_m:.3f} m")

    # Extract GT matches
    matches = extract_ground_truth_line_matches(view0, view1, target_scene)
    print(f"\nGround Truth Matches Found: {len(matches)} pairs")
    print("-" * 70)
    print(f"{'LineID':<8} | {'Semantic':<15} | {'Length View0':<14} | {'Length View1':<14} | {'Vis0':<6} | {'Vis1':<6}")
    print("-" * 70)
    for m in matches[:15]:
        print(
            f"{m.line_id:<8} | {m.semantic_name:<15} | {m.length0:>6.1f} px      | {m.length1:>6.1f} px      | {m.vis_ratio0:>4.2f} | {m.vis_ratio1:>4.2f}"
        )
    if len(matches) > 15:
        print(f"... and {len(matches) - 15} more line matches.")

    # Render visualization
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vis = draw_ground_truth_matches(view0, view1, matches, max_draw=args.max_draw)
    cv2.imwrite(str(args.output), vis)
    print(f"\nSaved Ground Truth matching visualization to: {args.output}")

    # Export machine-readable JSON
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    json_data = {
        "scene_id": target_scene.scene_id,
        "room_id": view0.room_id,
        "view0_id": view0.view_id,
        "view1_id": view1.view_id,
        "relative_rotation_deg": rot_deg,
        "baseline_meters": base_m,
        "total_gt_matches": len(matches),
        "matches": [
            {
                "line_id": m.line_id,
                "semantic": m.semantic_name,
                "line0": {"start": m.line0[0], "end": m.line0[1], "length": round(m.length0, 1)},
                "line1": {"start": m.line1[0], "end": m.line1[1], "length": round(m.length1, 1)},
                "visibility0": round(m.vis_ratio0, 3),
                "visibility1": round(m.vis_ratio1, 3),
            }
            for m in matches
        ],
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved machine-readable Ground Truth JSON to: {args.json_out}")


if __name__ == "__main__":
    main()
