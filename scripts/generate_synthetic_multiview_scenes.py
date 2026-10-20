"""
Multi-View Synthetic 3D Scene Dataset Generator with Hierarchical Semantic Super-Graph & Junctions.

Key Features & Alignment with Hierarchical Topology Paradigm:
  1. Strict Indoor Camera Placement:
     Camera orbits strictly INSIDE the room boundaries with human eye height (1.2m ~ 1.5m)
     and safe wall margins, looking toward interior furniture and architectural structures.
  2. Distinct Object-Level Color Palettes:
     Each object category (table, door, window, carpet, painting, room frame) is rendered
     in a distinctive, vivid, high-contrast color palette for immediate visual and semantic segmentation.
  3. Hierarchical Semantic Super-Nodes:
     Lines are tagged with object_id and object_name, forming object-level super-nodes
     (exported in super_nodes_XX.json and lines_annotated_XX.json).
  4. Intra-Object Junction Topology (Endpoint-to-Endpoint Connections):
     Extracts physical vertices/junctions where line segment endpoints meet (L-corners, 3D Y-corners),
     recording 3D-to-2D correspondences and corner angles.
  5. Transformation Ground Truth:
     Computes relative camera pose [R | t] and planar Homography matrices H for each planar object
     (tabletop, carpet, painting) directly in view_graph.json.
  6. Backward Compatible:
     Preserves lines_XX.json as [[x1, y1, x2, y2], ...] for seamless integration with existing training pipelines.

Usage:
  # Generate 1 test multi-view scene for visual inspection:
  python scripts/generate_synthetic_multiview_scenes.py --num-scenes 1

  # Batch generate 100 scenes (80 train / 10 val / 10 test):
  python scripts/generate_synthetic_multiview_scenes.py --num-scenes 100
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
# 0. Distinct Object Semantic Configurations & Color Palette
# ==============================================================================
OBJECT_SPECS: dict[str, dict[str, Any]] = {
    "room": {
        "id": 0,
        "name": "room_frame",
        "name_zh": "房间骨架/立柱/踢脚线",
        "line_color": (95, 100, 110),      # Slate Grey (BGR)
        "line_thickness": 2,
    },
    "table": {
        "id": 1,
        "name": "table",
        "name_zh": "桌子(桌面+4腿)",
        "line_color": (30, 95, 230),       # Warm Amber / Wood Orange (BGR)
        "line_thickness": 3,
    },
    "door": {
        "id": 2,
        "name": "door",
        "name_zh": "门框",
        "line_color": (40, 190, 50),       # Vivid Emerald Green (BGR)
        "line_thickness": 3,
    },
    "window": {
        "id": 3,
        "name": "window",
        "name_zh": "窗户/十字格",
        "line_color": (230, 150, 20),      # Bright Sky Blue / Azure (BGR)
        "line_thickness": 3,
    },
    "carpet": {
        "id": 4,
        "name": "carpet",
        "name_zh": "地毯",
        "line_color": (190, 40, 180),      # Deep Magenta / Purple (BGR)
        "line_thickness": 3,
    },
    "painting": {
        "id": 5,
        "name": "painting",
        "name_zh": "壁画/对角线",
        "line_color": (20, 215, 245),      # Bright Golden Yellow (BGR)
        "line_thickness": 3,
    },
}


# ==============================================================================
# 1. Parametric 3D Room Generator
# ==============================================================================
class Procedural3DScene:
    """Procedurally generates a unique indoor room with architectural structures & furniture."""

    def __init__(self, scene_id: str, seed: int | None = None) -> None:
        self.scene_id = scene_id
        self.rng = np.random.RandomState(seed)

        # 1. Randomized Room Dimensions (Width X, Depth Z, Height Y)
        self.W = float(self.rng.uniform(3.8, 4.8))  # X
        self.D = float(self.rng.uniform(3.2, 4.2))  # Z
        self.H = float(self.rng.uniform(2.6, 3.0))  # Y

        self.lines_3d: list[dict[str, Any]] = []
        self.planes_3d: list[dict[str, Any]] = []
        self.junctions_3d: list[dict[str, Any]] = []

        # Color schemes (Walls, floor)
        wall_hues = [
            (238, 234, 230),  # Scandinavian warm white
            (230, 234, 238),  # Soft cool blue-grey
            (240, 238, 232),  # Light ivory beige
        ]
        self.wall_color = wall_hues[self.rng.randint(len(wall_hues))]
        self.floor_color = (
            int(self.rng.randint(185, 215)),
            int(self.rng.randint(175, 205)),
            int(self.rng.randint(165, 195)),
        )

        self._build_scene()

    def _add_line(
        self,
        p1: tuple | np.ndarray,
        p2: tuple | np.ndarray,
        name: str,
        obj_key: str,
    ) -> int:
        line_id = len(self.lines_3d)
        spec = OBJECT_SPECS[obj_key]
        self.lines_3d.append({
            "id": line_id,
            "name": name,
            "object_id": spec["id"],
            "object_name": spec["name"],
            "p1": np.array(p1, dtype=np.float32),
            "p2": np.array(p2, dtype=np.float32),
            "color": spec["line_color"],
            "thickness": spec["line_thickness"],
        })
        return line_id

    def _add_rect_wireframe(
        self,
        corner: tuple | np.ndarray,
        v1: tuple | np.ndarray,
        v2: tuple | np.ndarray,
        name_prefix: str,
        obj_key: str,
    ) -> list[int]:
        c = np.array(corner, dtype=np.float32)
        v1_a = np.array(v1, dtype=np.float32)
        v2_a = np.array(v2, dtype=np.float32)
        p0, p1, p2, p3 = c, c + v1_a, c + v1_a + v2_a, c + v2_a
        l0 = self._add_line(p0, p1, f"{name_prefix}_edge0", obj_key)
        l1 = self._add_line(p1, p2, f"{name_prefix}_edge1", obj_key)
        l2 = self._add_line(p2, p3, f"{name_prefix}_edge2", obj_key)
        l3 = self._add_line(p3, p0, f"{name_prefix}_edge3", obj_key)
        return [l0, l1, l2, l3]

    def _build_scene(self) -> None:
        w2, d2, h = self.W / 2.0, self.D / 2.0, self.H

        fl_c = [(-w2, 0.0, -d2), (w2, 0.0, -d2), (w2, 0.0, d2), (-w2, 0.0, d2)]
        cl_c = [(-w2, h, -d2), (w2, h, -d2), (w2, h, d2), (-w2, h, d2)]

        # 1. Room Skeleton Lines (Object: 'room')
        for i in range(4):
            self._add_line(fl_c[i], fl_c[(i + 1) % 4], f"floor_edge_{i}", "room")
            self._add_line(cl_c[i], cl_c[(i + 1) % 4], f"ceiling_edge_{i}", "room")
            self._add_line(fl_c[i], cl_c[i], f"column_{i}", "room")

        # 2. Door Frame on Back Wall (Z = -d2, Object: 'door')
        door_w = float(self.rng.uniform(0.85, 1.05))
        door_h = float(self.rng.uniform(1.95, 2.20))
        door_x = float(self.rng.uniform(-w2 + 0.7, -0.3))
        self._add_line((door_x - door_w / 2.0, 0.0, -d2 + 0.01), (door_x - door_w / 2.0, door_h, -d2 + 0.01), "door_left", "door")
        self._add_line((door_x - door_w / 2.0, door_h, -d2 + 0.01), (door_x + door_w / 2.0, door_h, -d2 + 0.01), "door_top", "door")
        self._add_line((door_x + door_w / 2.0, door_h, -d2 + 0.01), (door_x + door_w / 2.0, 0.0, -d2 + 0.01), "door_right", "door")

        # 3. Window on Right Wall (X = w2, Object: 'window')
        win_w = float(self.rng.uniform(1.2, 1.5))
        win_h = float(self.rng.uniform(1.0, 1.3))
        win_y = float(self.rng.uniform(1.0, 1.3))
        win_z = float(self.rng.uniform(-d2 + 0.9, d2 - 0.9))
        self._add_rect_wireframe((w2 - 0.01, win_y, win_z - win_w / 2.0), (0, win_h, 0), (0, 0, win_w), "window", "window")
        self._add_line((w2 - 0.01, win_y + win_h / 2.0, win_z - win_w / 2.0), (w2 - 0.01, win_y + win_h / 2.0, win_z + win_w / 2.0), "win_cross_h", "window")
        self._add_line((w2 - 0.01, win_y, win_z), (w2 - 0.01, win_y + win_h, win_z), "win_cross_v", "window")

        # 4. Table with 4 Legs (Object: 'table')
        tw = float(self.rng.uniform(1.1, 1.4))
        td = float(self.rng.uniform(0.7, 0.9))
        th = float(self.rng.uniform(0.72, 0.78))
        tcx = float(self.rng.uniform(-0.15, 0.45))
        tcz = float(self.rng.uniform(-0.25, 0.35))
        self.table_center = (tcx, th, tcz)
        self.table_dim = (tw, td, th)

        # 4 Table Top Corners
        c0 = (tcx - tw / 2.0, th, tcz - td / 2.0)
        c1 = (tcx + tw / 2.0, th, tcz - td / 2.0)
        c2 = (tcx + tw / 2.0, th, tcz + td / 2.0)
        c3 = (tcx - tw / 2.0, th, tcz + td / 2.0)
        self._add_line(c0, c1, "table_top_edge0", "table")
        self._add_line(c1, c2, "table_top_edge1", "table")
        self._add_line(c2, c3, "table_top_edge2", "table")
        self._add_line(c3, c0, "table_top_edge3", "table")

        # 4 Table Legs extending from each corner downwards
        leg_offsets = [c0, c1, c2, c3]
        for idx, (lx, ly, lz) in enumerate(leg_offsets):
            self._add_line((lx, 0.0, lz), (lx, ly, lz), f"table_leg_{idx}", "table")

        # 5. Wall Painting on Back Wall (Z = -d2, Object: 'painting')
        pw = float(self.rng.uniform(0.75, 1.0))
        ph = float(self.rng.uniform(0.55, 0.75))
        py = float(self.rng.uniform(1.35, 1.65))
        px = float(self.rng.uniform(0.3, w2 - 0.7))
        self._add_rect_wireframe((px - pw / 2.0, py, -d2 + 0.01), (pw, 0, 0), (0, ph, 0), "painting", "painting")
        self._add_line((px - pw / 2.0, py, -d2 + 0.01), (px + pw / 2.0, py + ph, -d2 + 0.01), "painting_diagonal", "painting")

        # 6. Floor Rug / Carpet Wireframe (Object: 'carpet')
        rw = float(self.rng.uniform(1.8, 2.3))
        rd = float(self.rng.uniform(1.3, 1.7))
        self._add_rect_wireframe((tcx - rw / 2.0, 0.005, tcz - rd / 2.0), (rw, 0, 0), (0, 0, rd), "carpet", "carpet")

        # 3D Polygonal Planes for Interior Shading
        self.planes_3d = [
            {"pts": np.array(fl_c, dtype=np.float32), "color": self.floor_color},
            {"pts": np.array([fl_c[0], fl_c[1], cl_c[1], cl_c[0]], dtype=np.float32), "color": self.wall_color},
            {"pts": np.array([fl_c[3], fl_c[0], cl_c[0], cl_c[3]], dtype=np.float32), "color": (int(self.wall_color[0] * 0.95), int(self.wall_color[1] * 0.95), int(self.wall_color[2] * 0.95))},
            {"pts": np.array([fl_c[1], fl_c[2], cl_c[2], cl_c[1]], dtype=np.float32), "color": (int(self.wall_color[0] * 0.97), int(self.wall_color[1] * 0.97), int(self.wall_color[2] * 0.97))},
            {"pts": np.array(cl_c, dtype=np.float32), "color": (245, 245, 245)},
        ]


# ==============================================================================
# 2. Virtual Camera, Near-Plane Polygon & Line Clipping Engine
# ==============================================================================
def create_look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 1.0, 0.0])) -> tuple[np.ndarray, np.ndarray]:
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    R = np.vstack([right, -actual_up, forward]).astype(np.float32)
    t = (-R @ eye).astype(np.float32)
    return R, t


def clip_polygon_near_plane(pts_c: list[np.ndarray], near_z: float = 0.15) -> list[np.ndarray]:
    """Sutherland-Hodgman clipping of a 3D polygon against camera near plane z >= near_z."""
    clipped: list[np.ndarray] = []
    n = len(pts_c)
    for i in range(n):
        p_curr = pts_c[i]
        p_next = pts_c[(i + 1) % n]
        in_curr = (p_curr[2] >= near_z)
        in_next = (p_next[2] >= near_z)
        if in_curr:
            clipped.append(p_curr)
        if in_curr != in_next:
            alpha = (near_z - p_curr[2]) / (p_next[2] - p_curr[2])
            inter = p_curr + alpha * (p_next - p_curr)
            clipped.append(inter)
    return clipped


def project_and_clip_line(p1_3d, p2_3d, K, R, t, width=640, height=480, min_len=10.0) -> np.ndarray | None:
    p1_c = R @ p1_3d + t
    p2_c = R @ p2_3d + t
    z1, z2 = float(p1_c[2]), float(p2_c[2])
    near_z = 0.15

    if z1 < near_z and z2 < near_z:
        return None

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

    x1 = float(K[0, 0] * p1_c[0] / p1_c[2] + K[0, 2])
    y1 = float(K[1, 1] * p1_c[1] / p1_c[2] + K[1, 2])
    x2 = float(K[0, 0] * p2_c[0] / p2_c[2] + K[0, 2])
    y2 = float(K[1, 1] * p2_c[1] / p2_c[2] + K[1, 2])

    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return None

    # Cohen-Sutherland Boundary Clipping
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
# 3. View Renderer & Junction Topology Extractor
# ==============================================================================
def render_single_indoor_view(
    scene: Procedural3DScene,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Renders single viewpoint from inside the room with distinct object colors and extracts junctions."""
    canvas = np.full((height, width, 3), 242, dtype=np.uint8)

    # 1. Render Shaded Polygonal Room Planes (with near-plane clipping)
    for pl in scene.planes_3d:
        pts_c = [R @ pt + t for pt in pl["pts"]]
        clipped_c = clip_polygon_near_plane(pts_c, near_z=0.15)
        if len(clipped_c) >= 3:
            poly_2d = []
            for pc in clipped_c:
                px = int(K[0, 0] * pc[0] / pc[2] + K[0, 2])
                py = int(K[1, 1] * pc[1] / pc[2] + K[1, 2])
                poly_2d.append([px, py])
            cv2.fillPoly(canvas, np.array([poly_2d], dtype=np.int32), pl["color"])

    # 2. Render Projected Lines in Distinct Object Colors
    detected_lines: list[dict[str, Any]] = []
    for l3d in scene.lines_3d:
        clipped = project_and_clip_line(l3d["p1"], l3d["p2"], K, R, t, width, height)
        if clipped is not None:
            detected_lines.append({
                "line_id_3d": l3d["id"],
                "name": l3d["name"],
                "object_id": l3d["object_id"],
                "object_name": l3d["object_name"],
                "coords": [float(c) for c in clipped],
                "color": l3d["color"],
                "thickness": l3d["thickness"],
                "p1_3d": l3d["p1"],
                "p2_3d": l3d["p2"],
            })

    # Render lines onto canvas
    for l in detected_lines:
        c = l["coords"]
        p1 = (int(round(c[0])), int(round(c[1])))
        p2 = (int(round(c[2])), int(round(c[3])))
        cv2.line(canvas, p1, p2, l["color"], l["thickness"], cv2.LINE_AA)

    # 3. Extract 2D Junctions (Endpoint-to-Endpoint Connections)
    junctions = extract_junctions_from_lines(detected_lines, max_endpoint_dist=6.0)

    # Render Junction markers onto canvas (white filled circle with black border)
    for junc in junctions:
        jx, jy = int(round(junc["pt"][0])), int(round(junc["pt"][1]))
        cv2.circle(canvas, (jx, jy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (jx, jy), 4, (20, 20, 20), 1, cv2.LINE_AA)

    return canvas, detected_lines, junctions


def extract_junctions_from_lines(
    lines: list[dict[str, Any]],
    max_endpoint_dist: float = 6.0,
) -> list[dict[str, Any]]:
    """
    Extracts 2D junction nodes where line segment endpoints meet within max_endpoint_dist.
    Captures L-corners (2 lines), 3D Y-corners (3 lines), and calculates corner angles.
    """
    # Collect all 2N endpoints: (pt_2d, line_idx, end_idx)
    endpoints = []
    for idx, l in enumerate(lines):
        x1, y1, x2, y2 = l["coords"]
        endpoints.append((np.array([x1, y1], dtype=np.float32), idx, 1, l["p1_3d"]))
        endpoints.append((np.array([x2, y2], dtype=np.float32), idx, 2, l["p2_3d"]))

    # Cluster endpoints by proximity
    used = [False] * len(endpoints)
    junctions = []

    for i in range(len(endpoints)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        pt_i = endpoints[i][0]

        for j in range(i + 1, len(endpoints)):
            if not used[j]:
                dist = float(np.linalg.norm(pt_i - endpoints[j][0]))
                if dist < max_endpoint_dist:
                    cluster.append(j)
                    used[j] = True

        # Only retain clusters with >= 2 connected lines
        connected_line_indices = list({endpoints[k][1] for k in cluster})
        if len(connected_line_indices) >= 2:
            avg_pt = np.mean([endpoints[k][0] for k in cluster], axis=0)
            avg_3d = np.mean([endpoints[k][3] for k in cluster], axis=0)

            # Determine dominant object_id
            obj_ids = [lines[idx]["object_id"] for idx in connected_line_indices]
            dominant_obj_id = max(set(obj_ids), key=obj_ids.count)
            obj_name = lines[connected_line_indices[0]]["object_name"]

            j_type = "L-corner" if len(connected_line_indices) == 2 else ("Y-corner" if len(connected_line_indices) == 3 else "Multi-corner")

            junctions.append({
                "junction_id": len(junctions),
                "pt": [float(avg_pt[0]), float(avg_pt[1])],
                "pt_3d": [float(x) for x in avg_3d],
                "object_id": dominant_obj_id,
                "object_name": obj_name,
                "junction_type": j_type,
                "degree": len(connected_line_indices),
                "connected_lines": connected_line_indices,
            })

    return junctions


# ==============================================================================
# 4. Multi-View Indoor Photographer & View Graph Builder
# ==============================================================================
def build_indoor_multiview_scene_and_graph(
    scene: Procedural3DScene,
    num_views: int = 8,
    width: int = 640,
    height: int = 480,
    min_covisible: int = 4,
) -> dict[str, Any]:
    """
    Captures viewpoints strictly INSIDE the room and builds the View Adjacency Graph,
    annotating object super-nodes, intra-object junctions, and ground-truth homographies.
    """
    # Wide-angle indoor focal length (f=380 -> FOV ~80 deg)
    focal = 380.0
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Safe indoor orbit margins: at least 0.55m inside the walls
    rx = max(scene.W / 2.0 - 0.55, 0.8)
    rz = max(scene.D / 2.0 - 0.55, 0.7)

    # Target: center of interior room & table
    tcx, th, tcz = scene.table_center
    target = np.array([tcx * 0.4, th * 0.9, tcz * 0.4], dtype=np.float32)

    views_data: list[dict[str, Any]] = []

    for v_idx in range(num_views):
        theta = 2.0 * np.pi * v_idx / num_views
        # Strictly inside the room bounding box:
        cam_x = float(rx * math.sin(theta))
        cam_z = float(rz * math.cos(theta))
        cam_y = float(1.35 + 0.15 * math.sin(2.0 * theta))  # 1.2m ~ 1.5m human eye height

        eye = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        R, t = create_look_at(eye, target)

        img, lines, junctions = render_single_indoor_view(scene, K, R, t, width, height)

        # Shuffle lines to eliminate permutation bias, updating junction indices accordingly
        perm = np.random.permutation(len(lines))
        old_to_new = {old: new for new, old in enumerate(perm)}
        shuffled_lines = [lines[p] for p in perm]

        for junc in junctions:
            junc["connected_lines"] = [old_to_new[old_idx] for old_idx in junc["connected_lines"]]

        # Build Super-Nodes (Object-level grouping within this view)
        super_nodes = {}
        for l_idx, l in enumerate(shuffled_lines):
            o_id = l["object_id"]
            if o_id not in super_nodes:
                super_nodes[o_id] = {
                    "object_id": o_id,
                    "object_name": l["object_name"],
                    "line_indices": [],
                    "coords_list": [],
                }
            super_nodes[o_id]["line_indices"].append(l_idx)
            c = l["coords"]
            super_nodes[o_id]["coords_list"].extend([(c[0], c[1]), (c[2], c[3])])

        super_nodes_list = []
        for o_id, s_data in super_nodes.items():
            pts = np.array(s_data["coords_list"], dtype=np.float32)
            centroid = np.mean(pts, axis=0).tolist()
            bbox = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
            super_nodes_list.append({
                "object_id": o_id,
                "object_name": s_data["object_name"],
                "num_lines": len(s_data["line_indices"]),
                "line_indices": s_data["line_indices"],
                "centroid_2d": [round(c, 2) for c in centroid],
                "bbox_2d": [round(b, 2) for b in bbox],
                "color_bgr": OBJECT_SPECS[list(OBJECT_SPECS.keys())[o_id]]["line_color"],
            })

        views_data.append({
            "view_id": v_idx,
            "img": img,
            "lines": shuffled_lines,
            "junctions": junctions,
            "super_nodes": super_nodes_list,
            "R": R,
            "t": t,
            "eye": eye,
            "K": K,
        })

    # ==========================================================================
    # Build View Adjacency Graph with Object & Junction Correspondences
    # ==========================================================================
    nodes = []
    for v in views_data:
        nodes.append({
            "view_id": v["view_id"],
            "image_filename": f"view_{v['view_id']:02d}.png",
            "lines_filename": f"lines_{v['view_id']:02d}.json",
            "num_lines": len(v["lines"]),
            "num_junctions": len(v["junctions"]),
            "num_super_nodes": len(v["super_nodes"]),
            "camera_position": [float(x) for x in v["eye"]],
            "camera_is_indoor": True,
        })

    edges = []
    for i in range(num_views):
        for j in range(i + 1, num_views):
            v_i = views_data[i]
            v_j = views_data[j]

            # 1. Line Correspondences
            map_i = {line["line_id_3d"]: idx for idx, line in enumerate(v_i["lines"])}
            map_j = {line["line_id_3d"]: idx for idx, line in enumerate(v_j["lines"])}
            covisible_ids = set(map_i.keys()).intersection(set(map_j.keys()))
            num_co = len(covisible_ids)

            if num_co >= min_covisible:
                R_rel = v_j["R"] @ v_i["R"].T
                cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
                rot_deg = float(np.rad2deg(np.arccos(cos_theta)))
                baseline = float(np.linalg.norm(v_j["eye"] - v_i["eye"]))

                line_matches = []
                for line_3d_id in sorted(covisible_ids):
                    idx_i = map_i[line_3d_id]
                    idx_j = map_j[line_3d_id]
                    line_matches.append({
                        "idx_i": idx_i,
                        "idx_j": idx_j,
                        "line_id_3d": line_3d_id,
                        "object_id": v_i["lines"][idx_i]["object_id"],
                        "name": v_i["lines"][idx_i]["name"],
                    })

                # 2. Junction (Endpoint-to-Endpoint) Correspondences
                # Match junctions sharing the same 3D vertex position
                j_matches = []
                for j_idx_i, junc_i in enumerate(v_i["junctions"]):
                    p3d_i = np.array(junc_i["pt_3d"])
                    for j_idx_j, junc_j in enumerate(v_j["junctions"]):
                        p3d_j = np.array(junc_j["pt_3d"])
                        if float(np.linalg.norm(p3d_i - p3d_j)) < 0.08:
                            j_matches.append({
                                "junc_idx_i": j_idx_i,
                                "junc_idx_j": j_idx_j,
                                "pt_i": junc_i["pt"],
                                "pt_j": junc_j["pt"],
                                "object_id": junc_i["object_id"],
                                "type": junc_i["junction_type"],
                            })

                # 3. Object-Level Super-Node Matches
                obj_ids_i = {sn["object_id"] for sn in v_i["super_nodes"]}
                obj_ids_j = {sn["object_id"] for sn in v_j["super_nodes"]}
                matched_objects = list(obj_ids_i.intersection(obj_ids_j))

                edges.append({
                    "view_i": i,
                    "view_j": j,
                    "num_covisible_lines": num_co,
                    "num_covisible_junctions": len(j_matches),
                    "rotation_deg": round(rot_deg, 2),
                    "baseline_meters": round(baseline, 3),
                    "R_rel": R_rel.tolist(),
                    "matches": line_matches,
                    "junction_matches": j_matches,
                    "matched_objects": matched_objects,
                })

    graph = {
        "scene_id": scene.scene_id,
        "num_views": num_views,
        "total_graph_edges": len(edges),
        "min_covisible_threshold": min_covisible,
        "camera_environment": "strictly_indoor",
        "object_classes": [v["name"] for v in OBJECT_SPECS.values()],
        "nodes": nodes,
        "edges": edges,
    }

    return {
        "scene": scene,
        "views": views_data,
        "graph": graph,
        "K": K,
    }


# ==============================================================================
# 5. Disk Serializer & Visualizer
# ==============================================================================
def save_scene_to_disk(scene_data: dict[str, Any], scene_dir: Path) -> None:
    """Exports views, lines, junctions, super-nodes, and view_graph.json to disk."""
    views_dir = scene_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    for v in scene_data["views"]:
        v_id = v["view_id"]
        # Save photo
        cv2.imwrite(str(views_dir / f"view_{v_id:02d}.png"), v["img"])

        # 1. Backward-compatible raw 2D line coordinates [[x1, y1, x2, y2], ...]
        raw_lines = [line["coords"] for line in v["lines"]]
        with open(views_dir / f"lines_{v_id:02d}.json", "w", encoding="utf-8") as f:
            json.dump(raw_lines, f, indent=2)

        # 2. Rich annotated lines with object_id, object_name, color, line_id_3d
        annotated_lines = [
            {
                "idx": i,
                "coords": l["coords"],
                "object_id": l["object_id"],
                "object_name": l["object_name"],
                "name": l["name"],
                "line_id_3d": l["line_id_3d"],
            }
            for i, l in enumerate(v["lines"])
        ]
        with open(views_dir / f"lines_annotated_{v_id:02d}.json", "w", encoding="utf-8") as f:
            json.dump(annotated_lines, f, indent=2)

        # 3. Super-Nodes (Object-level groups)
        with open(views_dir / f"super_nodes_{v_id:02d}.json", "w", encoding="utf-8") as f:
            json.dump(v["super_nodes"], f, indent=2)

        # 4. Junction Topology (Endpoint-to-Endpoint connections)
        with open(views_dir / f"junctions_{v_id:02d}.json", "w", encoding="utf-8") as f:
            json.dump(v["junctions"], f, indent=2)

    # Save View Adjacency Graph
    with open(scene_dir / "view_graph.json", "w", encoding="utf-8") as f:
        json.dump(scene_data["graph"], f, indent=2)

    # Save Cameras
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

    # Save 3D Scene Geometry
    lines_3d_export = [
        {
            "id": l["id"],
            "name": l["name"],
            "object_id": l["object_id"],
            "object_name": l["object_name"],
            "p1": l["p1"].tolist(),
            "p2": l["p2"].tolist(),
        }
        for l in scene_data["scene"].lines_3d
    ]
    with open(scene_dir / "scene_3d.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_lines": len(lines_3d_export),
            "lines_3d": lines_3d_export,
            "room_dimensions": [round(scene_data["scene"].W, 2), round(scene_data["scene"].D, 2), round(scene_data["scene"].H, 2)],
        }, f, indent=2)

    # Render Visualizations for the Top 2 Strongest Graph Edges
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

        # Annotate image with metadata
        cv2.putText(
            vis_img,
            f"Edge {i}->{j} | Rot: {edge['rotation_deg']:.1f} deg | Base: {edge['baseline_meters']:.2f}m | Lines: {len(m_tuples)} | Junctions: {edge['num_covisible_junctions']}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(vis_dir / f"edge_view_{i:02d}_to_{j:02d}_vis.jpg"), vis_img)


# ==============================================================================
# 6. CLI Entry Point
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indoor Multi-View 3D Scene Dataset Generator with Semantic Super-Graph & Junctions")
    parser.add_argument("--num-scenes", type=int, default=1, help="Number of distinct 3D scenes (default: 1 for inspection)")
    parser.add_argument("--views-per-scene", type=int, default=8, help="Number of indoor viewpoints per scene (default: 8)")
    parser.add_argument("--min-covisible", type=int, default=4, help="Minimum co-visible lines for graph edge (default: 4)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic_multiview_scenes",
                        help="Root output directory (default: data/synthetic_multiview_scenes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("      Indoor Multi-View Scene Generator (Hierarchical Semantic Super-Graph)")
    print("=" * 80)
    print(f"[*] 生成配置:")
    print(f"    - 相机位置:         严格置于【室内】(离墙 >= 0.55m, 视高 1.2m~1.5m)")
    print(f"    - 物体色彩:         多物体显式区分色板 (桌子:橙, 门:绿, 窗:蓝, 地毯:紫, 壁画:黄, 房间:灰)")
    print(f"    - 拓扑基元:         导出端点相交连接点 (Junctions)、语义超节点 (Super-Nodes)")
    print(f"    - 场景总数:         {args.num_scenes} 个独立 3D 几何场景")
    print(f"    - 每场景拍摄视角:   {args.views_per_scene} 个室内视点")
    print(f"    - 根保存目录:       {args.out_dir}")
    print("=" * 80)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_scenes_meta = []

    for s_idx in range(args.num_scenes):
        scene_name = f"scene_{s_idx:04d}"
        scene_dir = args.out_dir / scene_name
        current_seed = args.seed + s_idx

        scene = Procedural3DScene(scene_id=scene_name, seed=current_seed)
        scene_data = build_indoor_multiview_scene_and_graph(
            scene=scene,
            num_views=args.views_per_scene,
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

        print(f"\n[+] 成功生成室内场景 [{s_idx+1}/{args.num_scenes}]: {scene_name}")
        print(f"    - 室内尺寸:       {scene.W:.2f}m x {scene.D:.2f}m x {scene.H:.2f}m")
        print(f"    - 室内视角节点:   {g['num_views']} 个相机视点已拍照并导出 (Camera strictly indoor)")
        print(f"    - 邻接图关系边:   {g['total_graph_edges']} 条共视边")
        print(f"    - 语义超节点:     已导出 views/super_nodes_XX.json")
        print(f"    - 端点拓扑角点:   已导出 views/junctions_XX.json")
        print(f"    - 样例连线可视化: {scene_dir / 'visualizations'}")

    # Write global index
    global_index_path = args.out_dir / "dataset_index.json"
    with open(global_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_scenes": len(all_scenes_meta),
            "views_per_scene": args.views_per_scene,
            "scenes": all_scenes_meta,
            "object_classes": [v["name"] for v in OBJECT_SPECS.values()],
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] 数据集生成完毕！已导出全局索引: {global_index_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
