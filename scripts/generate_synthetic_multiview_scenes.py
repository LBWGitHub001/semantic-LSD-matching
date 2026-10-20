"""
Multi-View Synthetic 3D Scene Dataset Generator with View Adjacency Graph.

Key Features:
  1. Multiple Diverse 3D Scenes:
     Generates distinct indoor architectural rooms (randomized room dimensions,
     furniture, window/door placements, and materials).
  2. Multi-View Photography per Scene:
     Captures multiple viewpoints (e.g. 6~8 camera poses in an orbit/room trajectory).
  3. View Adjacency Graph (view_graph.json):
     Computes graph connectivity between viewpoints based on co-visible 3D lines,
     recording relative rotations, baselines, and exact 1-to-1 line segment correspondences.
  4. Inspection & Batch Mode:
     Generates 1 multi-view scene first for user inspection and parameter tuning,
     ready for large-scale batch generation.

Usage:
  # Generate 1 multi-view scene (8 views + adjacency graph) for inspection:
  python scripts/generate_synthetic_multiview_scenes.py

  # Custom view count and camera trajectory:
  python scripts/generate_synthetic_multiview_scenes.py --views-per-scene 6 --min-covisible 4

  # Large-scale generation across 20 distinct scenes:
  python scripts/generate_synthetic_multiview_scenes.py --num-scenes 20 --out-dir data/multiview_scenes
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat.matcher import LineMatcher


# ==============================================================================
# 1. Parametric 3D Room Generator with Procedural Randomization
# ==============================================================================
class Procedural3DScene:
    """Procedurally generates a unique indoor room with architectural structures & furniture."""

    def __init__(self, scene_id: str, seed: int | None = None) -> None:
        self.scene_id = scene_id
        self.rng = np.random.RandomState(seed)

        # 1. Randomized Room Dimensions
        self.W = float(self.rng.uniform(3.4, 4.6))  # X
        self.D = float(self.rng.uniform(2.8, 3.8))  # Z
        self.H = float(self.rng.uniform(2.5, 3.0))  # Y

        self.lines_3d: list[dict[str, Any]] = []
        self.planes_3d: list[dict[str, Any]] = []

        # Color schemes (Walls, floor, wood, metals)
        wall_hues = [
            (235, 230, 225),  # Warm grey / ivory
            (225, 230, 235),  # Soft cool blue-grey
            (238, 235, 228),  # Scandinavian beige
        ]
        self.wall_color = wall_hues[self.rng.randint(len(wall_hues))]
        self.floor_color = (self.rng.randint(180, 215), self.rng.randint(170, 205), self.rng.randint(160, 195))

        self._build_scene()

    def _add_line(self, p1, p2, name: str, color=(35, 35, 35)) -> None:
        line_id = len(self.lines_3d)
        self.lines_3d.append({
            "id": line_id,
            "name": name,
            "p1": np.array(p1, dtype=np.float32),
            "p2": np.array(p2, dtype=np.float32),
            "color": color,
        })

    def _add_rect_wireframe(self, corner, v1, v2, name_prefix: str, color=(40, 40, 40)) -> None:
        c, v1, v2 = np.array(corner, dtype=np.float32), np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)
        p0, p1, p2, p3 = c, c + v1, c + v1 + v2, c + v2
        self._add_line(p0, p1, f"{name_prefix}_edge0", color)
        self._add_line(p1, p2, f"{name_prefix}_edge1", color)
        self._add_line(p2, p3, f"{name_prefix}_edge2", color)
        self._add_line(p3, p0, f"{name_prefix}_edge3", color)

    def _build_scene(self) -> None:
        w2, d2, h = self.W / 2.0, self.D / 2.0, self.H

        fl_c = [(-w2, 0.0, -d2), (w2, 0.0, -d2), (w2, 0.0, d2), (-w2, 0.0, d2)]
        cl_c = [(-w2, h, -d2), (w2, h, -d2), (w2, h, d2), (-w2, h, d2)]

        # 1. 12 Room Skeleton Lines
        for i in range(4):
            self._add_line(fl_c[i], fl_c[(i + 1) % 4], f"floor_edge_{i}", color=(50, 40, 30))
            self._add_line(cl_c[i], cl_c[(i + 1) % 4], f"ceiling_edge_{i}", color=(70, 70, 70))
            self._add_line(fl_c[i], cl_c[i], f"column_{i}", color=(55, 55, 55))

        # 2. Door Frame on Back Wall (Z = -d2)
        door_w, door_h = float(self.rng.uniform(0.85, 1.0)), float(self.rng.uniform(1.95, 2.2))
        door_x = float(self.rng.uniform(-w2 + 0.6, -0.2))
        self._add_line((door_x - door_w / 2.0, 0.0, -d2 + 0.01), (door_x - door_w / 2.0, door_h, -d2 + 0.01), "door_left", (90, 45, 20))
        self._add_line((door_x - door_w / 2.0, door_h, -d2 + 0.01), (door_x + door_w / 2.0, door_h, -d2 + 0.01), "door_top", (90, 45, 20))
        self._add_line((door_x + door_w / 2.0, door_h, -d2 + 0.01), (door_x + door_w / 2.0, 0.0, -d2 + 0.01), "door_right", (90, 45, 20))

        # 3. Window on Right Wall (X = w2)
        win_w, win_h = float(self.rng.uniform(1.1, 1.4)), float(self.rng.uniform(1.0, 1.3))
        win_y = float(self.rng.uniform(0.9, 1.2))
        win_z = float(self.rng.uniform(-d2 + 0.8, d2 - 0.8))
        self._add_rect_wireframe((w2 - 0.01, win_y, win_z - win_w / 2.0), (0, win_h, 0), (0, 0, win_w), "window", (20, 80, 150))
        self._add_line((w2 - 0.01, win_y + win_h / 2.0, win_z - win_w / 2.0), (w2 - 0.01, win_y + win_h / 2.0, win_z + win_w / 2.0), "win_cross_h", (20, 80, 150))
        self._add_line((w2 - 0.01, win_y, win_z), (w2 - 0.01, win_y + win_h, win_z), "win_cross_v", (20, 80, 150))

        # 4. Table with 4 Legs
        tw, td, th = float(self.rng.uniform(1.0, 1.3)), float(self.rng.uniform(0.65, 0.85)), float(self.rng.uniform(0.70, 0.78))
        tcx, tcz = float(self.rng.uniform(-0.2, 0.6)), float(self.rng.uniform(-0.3, 0.5))
        self._add_rect_wireframe((tcx - tw / 2.0, th, tcz - td / 2.0), (tw, 0, 0), (0, 0, td), "table_top", (110, 60, 25))
        for idx, (lx, lz) in enumerate([
            (tcx - tw / 2.0 + 0.04, tcz - td / 2.0 + 0.04),
            (tcx + tw / 2.0 - 0.04, tcz - td / 2.0 + 0.04),
            (tcx + tw / 2.0 - 0.04, tcz + td / 2.0 - 0.04),
            (tcx - tw / 2.0 + 0.04, tcz + td / 2.0 - 0.04),
        ]):
            self._add_line((lx, 0.0, lz), (lx, th, lz), f"table_leg_{idx}", (80, 45, 20))

        # 5. Wall Painting on Back Wall
        pw, ph, py, px = float(self.rng.uniform(0.7, 0.9)), float(self.rng.uniform(0.5, 0.7)), float(self.rng.uniform(1.3, 1.6)), float(self.rng.uniform(0.4, w2 - 0.6))
        self._add_rect_wireframe((px - pw / 2.0, py, -d2 + 0.01), (pw, 0, 0), (0, ph, 0), "painting", (150, 30, 30))
        self._add_line((px - pw / 2.0, py, -d2 + 0.01), (px + pw / 2.0, py + ph, -d2 + 0.01), "painting_diagonal", (170, 90, 30))

        # 6. Floor Rug / Carpet Wireframe
        rw, rd = float(self.rng.uniform(1.6, 2.2)), float(self.rng.uniform(1.2, 1.6))
        rcx, rcz = tcx, tcz
        self._add_rect_wireframe((rcx - rw / 2.0, 0.005, rcz - rd / 2.0), (rw, 0, 0), (0, 0, rd), "carpet", (60, 90, 90))

        # 3D Polygonal Planes for Shading
        self.planes_3d = [
            {"pts": np.array(fl_c, dtype=np.float32), "color": self.floor_color},
            {"pts": np.array([fl_c[0], fl_c[1], cl_c[1], cl_c[0]], dtype=np.float32), "color": self.wall_color},
            {"pts": np.array([fl_c[3], fl_c[0], cl_c[0], cl_c[3]], dtype=np.float32), "color": (int(self.wall_color[0] * 0.95), int(self.wall_color[1] * 0.95), int(self.wall_color[2] * 0.95))},
            {"pts": np.array([fl_c[1], fl_c[2], cl_c[2], cl_c[1]], dtype=np.float32), "color": (int(self.wall_color[0] * 0.97), int(self.wall_color[1] * 0.97), int(self.wall_color[2] * 0.97))},
            {"pts": np.array(cl_c, dtype=np.float32), "color": (245, 245, 245)},
        ]


# ==============================================================================
# 2. Virtual Camera & Frustum Projection Engine
# ==============================================================================
def create_look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 1.0, 0.0])) -> tuple[np.ndarray, np.ndarray]:
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    R = np.vstack([right, -actual_up, forward]).astype(np.float32)
    t = (-R @ eye).astype(np.float32)
    return R, t


def project_and_clip_line(p1_3d, p2_3d, K, R, t, width=640, height=480, min_len=12.0) -> np.ndarray | None:
    p1_c = R @ p1_3d + t
    p2_c = R @ p2_3d + t
    z1, z2 = float(p1_c[2]), float(p2_c[2])
    near_z = 0.15

    # If both endpoints are behind near plane, line is completely invisible
    if z1 < near_z and z2 < near_z:
        return None

    # Clip 3D line against near plane z = near_z
    if z1 < near_z:
        if abs(z2 - z1) < 1e-6:
            return None
        alpha = (near_z - z1) / (z2 - z1)
        p1_c = (1.0 - alpha) * p1_c + alpha * p2_c
    elif z2 < near_z:
        if abs(z1 - z2) < 1e-6:
            return None
        alpha = (near_z - z2) / (z1 - z2)
        p2_c = (1.0 - alpha) * p2_c + alpha * p1_c

    # Project to 2D
    x1 = float(K[0, 0] * p1_c[0] / p1_c[2] + K[0, 2])
    y1 = float(K[1, 1] * p1_c[1] / p1_c[2] + K[1, 2])
    x2 = float(K[0, 0] * p2_c[0] / p2_c[2] + K[0, 2])
    y2 = float(K[1, 1] * p2_c[1] / p2_c[2] + K[1, 2])

    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return None

    # 2D Cohen-Sutherland Boundary Clipping
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def get_code(x, y):
        code = INSIDE
        if x < 0:
            code |= LEFT
        elif x > width - 1:
            code |= RIGHT
        if y < 0:
            code |= TOP
        elif y > height - 1:
            code |= BOTTOM
        return code

    c1, c2 = get_code(x1, y1), get_code(x2, y2)
    for _ in range(8):
        if (c1 == 0) and (c2 == 0):
            break
        elif (c1 & c2) != 0:
            return None
        else:
            c_out = c1 if c1 != 0 else c2
            if c_out & TOP:
                dy = y2 - y1
                if abs(dy) < 1e-6:
                    return None
                x = x1 + (x2 - x1) * (0.0 - y1) / dy
                y = 0.0
            elif c_out & BOTTOM:
                dy = y2 - y1
                if abs(dy) < 1e-6:
                    return None
                x = x1 + (x2 - x1) * (float(height - 1) - y1) / dy
                y = float(height - 1)
            elif c_out & RIGHT:
                dx = x2 - x1
                if abs(dx) < 1e-6:
                    return None
                y = y1 + (y2 - y1) * (float(width - 1) - x1) / dx
                x = float(width - 1)
            elif c_out & LEFT:
                dx = x2 - x1
                if abs(dx) < 1e-6:
                    return None
                y = y1 + (y2 - y1) * (0.0 - x1) / dx
                x = 0.0

            if c_out == c1:
                x1, y1 = x, y
                c1 = get_code(x1, y1)
            else:
                x2, y2 = x, y
                c2 = get_code(x2, y2)

    length = math.hypot(x2 - x1, y2 - y1)
    if length < min_len:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ==============================================================================
# 3. Multi-View Scene Photographer & Adjacency Graph Builder
# ==============================================================================
def render_single_view(scene: Procedural3DScene, K, R, t, width=640, height=480) -> tuple[np.ndarray, list[dict]]:
    canvas = np.full((height, width, 3), 242, dtype=np.uint8)

    # 1. Shaded Planes
    for pl in scene.planes_3d:
        poly = []
        valid = True
        for pt in pl["pts"]:
            pc = R @ pt + t
            if pc[2] <= 0.05:
                valid = False
                break
            px = int(K[0, 0] * pc[0] / pc[2] + K[0, 2])
            py = int(K[1, 1] * pc[1] / pc[2] + K[1, 2])
            poly.append([px, py])
        if valid and len(poly) >= 3:
            cv2.fillPoly(canvas, np.array([poly], dtype=np.int32), pl["color"])

    # 2. Projected Lines
    detected_lines = []
    for l3d in scene.lines_3d:
        clipped = project_and_clip_line(l3d["p1"], l3d["p2"], K, R, t, width, height)
        if clipped is not None:
            detected_lines.append({
                "line_id_3d": l3d["id"],
                "name": l3d["name"],
                "coords": [float(c) for c in clipped],
                "color": l3d["color"],
            })
            p1 = (int(round(clipped[0])), int(round(clipped[1])))
            p2 = (int(round(clipped[2])), int(round(clipped[3])))
            cv2.line(canvas, p1, p2, l3d["color"], 2, cv2.LINE_AA)

    return canvas, detected_lines


def build_multiview_scene_and_graph(
    scene: Procedural3DScene,
    num_views: int = 8,
    radius: float = 3.6,
    width: int = 640,
    height: int = 480,
    min_covisible: int = 4,
) -> dict[str, Any]:
    """Captures N viewpoints around the 3D scene and builds the View Adjacency Graph."""
    focal = 550.0
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    target = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    views_data: list[dict[str, Any]] = []

    # Generate viewpoints along a 360-degree orbit around room center
    for v_idx in range(num_views):
        angle = 2.0 * np.pi * v_idx / num_views
        # Slight elevation oscillation to create natural 3D viewpoints
        elev = np.deg2rad(18.0 + 6.0 * np.sin(2.0 * angle))
        cam_y = 1.0 + radius * math.sin(elev)
        cam_xz = radius * math.cos(elev)

        eye = np.array([cam_xz * math.sin(angle), cam_y, cam_xz * math.cos(angle)], dtype=np.float32)
        R, t = create_look_at(eye, target)

        img, lines = render_single_view(scene, K, R, t, width, height)

        # Shuffle line ordering in 2D array to strictly eliminate order bias
        perm = np.random.permutation(len(lines))
        shuffled_lines = [lines[p] for p in perm]

        views_data.append({
            "view_id": v_idx,
            "img": img,
            "lines": shuffled_lines,
            "R": R,
            "t": t,
            "eye": eye,
            "K": K,
        })

    # ==========================================================================
    # Build View Adjacency Graph (Nodes & Edges)
    # ==========================================================================
    nodes = []
    for v in views_data:
        nodes.append({
            "view_id": v["view_id"],
            "image_filename": f"view_{v['view_id']:02d}.png",
            "lines_filename": f"lines_{v['view_id']:02d}.json",
            "num_lines": len(v["lines"]),
            "camera_position": [float(x) for x in v["eye"]],
        })

    edges = []
    for i in range(num_views):
        for j in range(i + 1, num_views):
            v_i = views_data[i]
            v_j = views_data[j]

            # 3D line map to 2D indices
            map_i = {line["line_id_3d"]: idx for idx, line in enumerate(v_i["lines"])}
            map_j = {line["line_id_3d"]: idx for idx, line in enumerate(v_j["lines"])}

            covisible_ids = set(map_i.keys()).intersection(set(map_j.keys()))
            num_co = len(covisible_ids)

            if num_co >= min_covisible:
                # Relative rotation in degrees
                R_rel = v_j["R"] @ v_i["R"].T
                cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
                rot_deg = float(np.rad2deg(np.arccos(cos_theta)))
                baseline = float(np.linalg.norm(v_j["eye"] - v_i["eye"]))

                matches = []
                for line_3d_id in sorted(covisible_ids):
                    idx_i = map_i[line_3d_id]
                    idx_j = map_j[line_3d_id]
                    matches.append({
                        "idx_i": idx_i,
                        "idx_j": idx_j,
                        "line_id_3d": line_3d_id,
                        "name": v_i["lines"][idx_i]["name"],
                    })

                edges.append({
                    "view_i": i,
                    "view_j": j,
                    "num_covisible_lines": num_co,
                    "rotation_deg": round(rot_deg, 2),
                    "baseline_meters": round(baseline, 3),
                    "overlap_ratio_i": round(num_co / max(len(v_i["lines"]), 1), 3),
                    "overlap_ratio_j": round(num_co / max(len(v_j["lines"]), 1), 3),
                    "matches": matches,
                })

    graph = {
        "scene_id": scene.scene_id,
        "num_views": num_views,
        "total_graph_edges": len(edges),
        "min_covisible_threshold": min_covisible,
        "nodes": nodes,
        "edges": edges,
    }

    return {
        "scene": scene,
        "views": views_data,
        "graph": graph,
        "K": K,
    }


def save_scene_to_disk(scene_data: dict[str, Any], scene_dir: Path) -> None:
    """Exports all viewpoints, line jsons, and the view_graph.json to disk."""
    views_dir = scene_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Views Images & 2D Lines
    for v in scene_data["views"]:
        v_id = v["view_id"]
        cv2.imwrite(str(views_dir / f"view_{v_id:02d}.png"), v["img"])

        lines_coords = [line["coords"] for line in v["lines"]]
        with open(views_dir / f"lines_{v_id:02d}.json", "w", encoding="utf-8") as f:
            json.dump(lines_coords, f, indent=2)

    # 2. Save View Adjacency Graph JSON (The Core Relationship Map)
    graph_path = scene_dir / "view_graph.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(scene_data["graph"], f, indent=2)

    # 3. Save Cameras Extrinsics / Intrinsics
    cameras_data = {
        "K": scene_data["K"].tolist(),
        "views": {
            f"view_{v['view_id']:02d}": {
                "R": v["R"].tolist(),
                "t": v["t"].tolist(),
                "eye": v["eye"].tolist(),
            }
            for v in scene_data["views"]
        },
    }
    with open(scene_dir / "cameras.json", "w", encoding="utf-8") as f:
        json.dump(cameras_data, f, indent=2)

    # 4. Save Raw 3D Scene Geometry
    lines_3d_export = [
        {"id": l["id"], "name": l["name"], "p1": l["p1"].tolist(), "p2": l["p2"].tolist()}
        for l in scene_data["scene"].lines_3d
    ]
    with open(scene_dir / "scene_3d.json", "w", encoding="utf-8") as f:
        json.dump({"total_lines": len(lines_3d_export), "lines_3d": lines_3d_export}, f, indent=2)

    # 5. Render Sample Visualizations for the First 2 Strongest Graph Edges
    vis_dir = scene_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sorted_edges = sorted(scene_data["graph"]["edges"], key=lambda e: e["num_covisible_lines"], reverse=True)
    for edge_rank, edge in enumerate(sorted_edges[:2]):
        i, j = edge["view_i"], edge["view_j"]
        v_i = scene_data["views"][i]
        v_j = scene_data["views"][j]

        l0_arr = np.array([l["coords"] for l in v_i["lines"]], dtype=np.float32)
        l1_arr = np.array([l["coords"] for l in v_j["lines"]], dtype=np.float32)
        m_tuples = [(m["idx_i"], m["idx_j"], 1.0) for m in edge["matches"]]

        vis_img = LineMatcher.draw_matches(v_i["img"], v_j["img"], l0_arr, l1_arr, m_tuples, max_draw=len(m_tuples))
        cv2.imwrite(str(vis_dir / f"edge_view_{i:02d}_to_{j:02d}_vis.jpg"), vis_img)


# ==============================================================================
# 4. CLI Argument Parser & Entry Point
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-View 3D Scene Dataset Generator with View Adjacency Graph")
    parser.add_argument("--num-scenes", type=int, default=1, help="Number of distinct 3D scenes (default: 1 for inspection)")
    parser.add_argument("--views-per-scene", type=int, default=8, help="Number of viewpoints per scene (default: 8)")
    parser.add_argument("--min-covisible", type=int, default=4, help="Minimum co-visible 3D lines to form an edge in graph (default: 4)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic_multiview_scenes",
                        help="Root output directory (default: data/synthetic_multiview_scenes)")
    parser.add_argument("--radius", type=float, default=3.6, help="Camera orbit radius (default: 3.6m)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("      Multi-View 3D Scene & View Adjacency Graph Dataset Generator")
    print("=" * 80)
    print(f"[*] 规划配置:")
    print(f"    - 场景总数:         {args.num_scenes} 个独立 3D 几何场景")
    print(f"    - 每场景拍摄视角数: {args.views_per_scene} 个环视视点")
    print(f"    - 邻接边建连阈值:   共视线段数 >= {args.min_covisible} 根")
    print(f"    - 根保存目录:       {args.out_dir}")
    print("=" * 80)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_scenes_meta = []

    for s_idx in range(args.num_scenes):
        scene_name = f"scene_{s_idx:04d}"
        scene_dir = args.out_dir / scene_name
        current_seed = args.seed + s_idx

        scene = Procedural3DScene(scene_id=scene_name, seed=current_seed)
        scene_data = build_multiview_scene_and_graph(
            scene=scene,
            num_views=args.views_per_scene,
            radius=args.radius,
            min_covisible=args.min_covisible,
        )

        save_scene_to_disk(scene_data, scene_dir)

        g = scene_data["graph"]
        all_scenes_meta.append({
            "scene_id": scene_name,
            "scene_dir": str(scene_dir.relative_to(ROOT)),
            "num_views": g["num_views"],
            "total_edges": g["total_graph_edges"],
            "room_size": [round(scene.W, 2), round(scene.D, 2), round(scene.H, 2)],
        })

        print(f"\n[+] 成功生成场景 [{s_idx+1}/{args.num_scenes}]: {scene_name}")
        print(f"    - 3D 房间长宽高:   {scene.W:.2f}m x {scene.D:.2f}m x {scene.H:.2f}m")
        print(f"    - 视角节点 (Nodes): {g['num_views']} 个相机视点已拍照并导出")
        print(f"    - 邻接边 (Edges):   {g['total_graph_edges']} 条具有充分共视线段的匹配边")
        print(f"    - 邻接图关系文件:   {scene_dir / 'view_graph.json'}")
        print(f"    - 样例连线大图:     {scene_dir / 'visualizations'}")

    # Write global dataset index
    global_index_path = args.out_dir / "dataset_index.json"
    with open(global_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_scenes": len(all_scenes_meta),
            "views_per_scene": args.views_per_scene,
            "scenes": all_scenes_meta,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] 数据集生成完毕！已导出全局索引: {global_index_path}")
    print(f"💡 提示: 您可以查看 `{args.out_dir}/scene_0000/view_graph.json` 及其渲染结果进行微调。")
    print(f"   满意后，直接增加 `--num-scenes 50` 即可批量生成 50 个多样化多视角场景！")
    print("=" * 80)


if __name__ == "__main__":
    main()
