"""
Unified Multi-Modal Dataset Generator Factory with Pluggable Components.

Extensible Component-Based Factory Architecture:
  1. BaseDatasetComponent: Lifecycle interface for all modality generators.
  2. ObjectMaskComponent: Instance masks, bounding boxes, and cross-view object correspondences.
  3. LSDLineComponent: Straight line segment detection (LSD), geometric descriptors,
     semantic region association, and cross-view line matching ground truth.
  4. Wireframe3DLineComponent: Analytical 3D CAD wireframe edge projections and camera poses.
  5. DatasetGeneratorFactory: Unified builder and execution manager.

Usage:
  # 1. Generate dataset containing straight lines and semantic objects (default):
  python generate_dataset.py --task semantic_line --num-scenes 50 --out-dir data/synthetic_semantic_lines

  # 2. Generate pure object matching dataset:
  python generate_dataset.py --task object --num-scenes 50

  # 3. Generate pure line matching dataset:
  python generate_dataset.py --task line --num-scenes 50

  # 4. Programmatic usage in Python:
  from generate_dataset import DatasetGeneratorFactory
  factory = DatasetGeneratorFactory.create(task="semantic_line", num_scenes=10)
  factory.generate(out_dir="data/my_line_dataset")
"""

from __future__ import annotations

import abc
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================================================================
# 0. 3D Math, Virtual Camera & Near-Plane Clipping
# ==============================================================================
def create_look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 1.0, 0.0])) -> tuple[np.ndarray, np.ndarray]:
    """Look-At camera extrinsics: X_c = R * X_w + t"""
    z_axis = target - eye
    z_norm = np.linalg.norm(z_axis)
    z_axis = z_axis / (z_norm + 1e-8)

    x_axis = np.cross(z_axis, up)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-4:
        up_alt = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(z_axis, up_alt)
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / (x_norm + 1e-8)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    R = np.vstack([x_axis, y_axis, z_axis])
    t = -R @ eye
    return R, t


def clip_polygon_near_plane(pts_c: list[np.ndarray], near_z: float = 0.15) -> list[np.ndarray]:
    """Sutherland-Hodgman clipping against near plane z = near_z."""
    clipped = []
    n = len(pts_c)
    if n == 0:
        return clipped
    for i in range(n):
        p1 = pts_c[i]
        p2 = pts_c[(i + 1) % n]
        in1 = p1[2] >= near_z
        in2 = p2[2] >= near_z
        if in1 and in2:
            clipped.append(p2)
        elif in1 and not in2:
            alpha = (near_z - p1[2]) / (p2[2] - p1[2] + 1e-9)
            clipped.append(p1 + alpha * (p2 - p1))
        elif not in1 and in2:
            alpha = (near_z - p1[2]) / (p2[2] - p1[2] + 1e-9)
            clipped.append(p1 + alpha * (p2 - p1))
            clipped.append(p2)
    return clipped


def compute_polygon_normal(pts: np.ndarray) -> np.ndarray:
    """Computes unit normal vector for a 3D planar polygon."""
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm > 1e-6:
        return n / norm
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def apply_shading(color_bgr: tuple[int, int, int], normal_w: np.ndarray, light_dir_w: np.ndarray) -> tuple[int, int, int]:
    """Applies diffuse + ambient directional shading."""
    cos_theta = max(0.0, float(np.dot(normal_w, light_dir_w)))
    intensity = 0.75 + 0.25 * cos_theta
    b = int(np.clip(color_bgr[0] * intensity, 0, 255))
    g = int(np.clip(color_bgr[1] * intensity, 0, 255))
    r = int(np.clip(color_bgr[2] * intensity, 0, 255))
    return (b, g, r)


# ==============================================================================
# 1. Procedural 3D Room Generator with Dimension 2 Hardening
# ==============================================================================
class UnifiedProceduralRoom:
    """
    Procedural 3D indoor room with structural wireframe lines and furniture:
      - Dimension 2 randomized material palettes & diffuse directional shading.
      - Globally indexed 3D physical edges for ground-truth line matching.
    """

    def __init__(self, seed: int | None = None, random_colors: bool = True) -> None:
        self.rng = np.random.RandomState(seed)
        self.random_colors = random_colors

        self.W = float(self.rng.uniform(3.5, 4.4))
        self.D = float(self.rng.uniform(3.5, 4.4))
        self.H = float(self.rng.uniform(2.6, 2.9))

        self.light_dir = np.array([0.4, 0.8, -0.4], dtype=np.float32)
        self.light_dir /= np.linalg.norm(self.light_dir)

        self._init_palettes()
        self.objects: list[dict[str, Any]] = []
        self._build_geometry()

    def _random_bgr(self, min_val: int = 40, max_val: int = 230) -> tuple[int, int, int]:
        return (
            int(self.rng.uniform(min_val, max_val)),
            int(self.rng.uniform(min_val, max_val)),
            int(self.rng.uniform(min_val, max_val)),
        )

    def _init_palettes(self) -> None:
        if not self.random_colors:
            self.floor_color = (200, 200, 200)
            self.wall_color = (235, 235, 235)
            self.table_color = (35, 110, 245)
            self.table_line = (30, 95, 230)
            self.door_color = (50, 205, 70)
            self.door_line = (40, 190, 50)
            self.win_color = (245, 175, 45)
            self.win_line = (230, 150, 20)
            self.carpet_color = (210, 60, 200)
            self.carpet_line = (190, 40, 180)
            self.paint_color = (30, 225, 250)
            self.paint_line = (20, 215, 245)
        else:
            wb = int(self.rng.uniform(215, 245))
            self.wall_color = (wb, int(wb * self.rng.uniform(0.97, 1.0)), int(wb * self.rng.uniform(0.95, 1.0)))
            fb = int(self.rng.uniform(160, 210))
            self.floor_color = (fb, int(fb * self.rng.uniform(0.92, 1.02)), int(fb * self.rng.uniform(0.88, 1.0)))

            table_types = [
                ((40, 75, 130), (30, 60, 110)),
                ((70, 130, 180), (55, 110, 160)),
                ((30, 30, 35), (15, 15, 20)),
                ((180, 190, 200), (150, 160, 170)),
                ((110, 100, 90), (90, 80, 70)),
            ]
            self.table_color, self.table_line = table_types[self.rng.randint(len(table_types))]

            door_types = [
                ((220, 225, 230), (170, 175, 180)),
                ((30, 40, 90), (20, 30, 75)),
                ((85, 90, 95), (65, 70, 75)),
            ]
            self.door_color, self.door_line = door_types[self.rng.randint(len(door_types))]

            win_types = [
                ((230, 230, 235), (180, 180, 190)),
                ((35, 35, 40), (20, 20, 25)),
            ]
            self.win_color, self.win_line = win_types[self.rng.randint(len(win_types))]

            c_fill = self._random_bgr(50, 200)
            self.carpet_color, self.carpet_line = c_fill, (max(0, c_fill[0] - 25), max(0, c_fill[1] - 25), max(0, c_fill[2] - 25))
            p_fill = self._random_bgr(60, 220)
            self.paint_color, self.paint_line = p_fill, (max(0, p_fill[0] - 20), max(0, p_fill[1] - 20), max(0, p_fill[2] - 20))

    def _build_geometry(self) -> None:
        w2, d2, h = self.W / 2.0, self.D / 2.0, self.H

        # 1. Room Shell
        fl_c = [(-w2, 0.0, -d2), (w2, 0.0, -d2), (w2, 0.0, d2), (-w2, 0.0, d2)]
        cl_c = [(-w2, h, -d2), (w2, h, -d2), (w2, h, d2), (-w2, h, d2)]

        self.planes_room = [
            {"pts": np.array(fl_c, dtype=np.float32), "color": self.floor_color, "normal": np.array([0, 1, 0], dtype=np.float32), "name": "floor"},
            {"pts": np.array([fl_c[0], fl_c[1], cl_c[1], cl_c[0]], dtype=np.float32), "color": self.wall_color, "normal": np.array([0, 0, 1], dtype=np.float32), "name": "wall_back"},
            {"pts": np.array([fl_c[3], fl_c[0], cl_c[0], cl_c[3]], dtype=np.float32), "color": self.wall_color, "normal": np.array([1, 0, 0], dtype=np.float32), "name": "wall_left"},
            {"pts": np.array([fl_c[1], fl_c[2], cl_c[2], cl_c[1]], dtype=np.float32), "color": self.wall_color, "normal": np.array([-1, 0, 0], dtype=np.float32), "name": "wall_right"},
            {"pts": np.array(cl_c, dtype=np.float32), "color": (245, 245, 245), "normal": np.array([0, -1, 0], dtype=np.float32), "name": "ceiling"},
        ]

        self.lines_room = []
        for i in range(4):
            self.lines_room.append((fl_c[i], fl_c[(i + 1) % 4]))
            self.lines_room.append((cl_c[i], cl_c[(i + 1) % 4]))
            self.lines_room.append((fl_c[i], cl_c[i]))

        # 2. Table (global_id = 1)
        tw = float(self.rng.uniform(1.1, 1.4))
        td = float(self.rng.uniform(0.7, 0.9))
        th = float(self.rng.uniform(0.72, 0.78))
        tcx = float(self.rng.uniform(-0.2, 0.45))
        tcz = float(self.rng.uniform(-0.25, 0.35))
        c0 = (tcx - tw / 2.0, th, tcz - td / 2.0)
        c1 = (tcx + tw / 2.0, th, tcz - td / 2.0)
        c2 = (tcx + tw / 2.0, th, tcz + td / 2.0)
        c3 = (tcx - tw / 2.0, th, tcz + td / 2.0)
        table_top_poly = np.array([c0, c1, c2, c3], dtype=np.float32)
        table_lines = [
            (c0, c1), (c1, c2), (c2, c3), (c3, c0),
            (c0, (c0[0], 0.0, c0[2])), (c1, (c1[0], 0.0, c1[2])),
            (c2, (c2[0], 0.0, c2[2])), (c3, (c3[0], 0.0, c3[2])),
        ]
        self.objects.append({
            "global_id": 1,
            "category": "table",
            "polys": [{"pts": table_top_poly, "color": self.table_color, "normal": np.array([0, 1, 0], dtype=np.float32)}],
            "lines": table_lines,
            "color_bgr": self.table_line,
        })

        # 3. Door (global_id = 2) on Back Wall (Z = -d2)
        door_w = float(self.rng.uniform(0.85, 1.05))
        door_h = float(self.rng.uniform(1.95, 2.20))
        door_x = float(self.rng.uniform(-w2 + 0.7, -0.3))
        d_p0 = (door_x - door_w / 2.0, 0.0, -d2 + 0.01)
        d_p1 = (door_x + door_w / 2.0, 0.0, -d2 + 0.01)
        d_p2 = (door_x + door_w / 2.0, door_h, -d2 + 0.01)
        d_p3 = (door_x - door_w / 2.0, door_h, -d2 + 0.01)
        door_poly = np.array([d_p0, d_p1, d_p2, d_p3], dtype=np.float32)
        self.objects.append({
            "global_id": 2,
            "category": "door",
            "polys": [{"pts": door_poly, "color": self.door_color, "normal": np.array([0, 0, 1], dtype=np.float32)}],
            "lines": [(d_p0, d_p3), (d_p3, d_p2), (d_p2, d_p1)],
            "color_bgr": self.door_line,
        })

        # 4. Window (global_id = 3) on Right Wall (X = w2)
        win_w = float(self.rng.uniform(1.2, 1.5))
        win_h = float(self.rng.uniform(1.0, 1.3))
        win_y = float(self.rng.uniform(1.0, 1.3))
        win_z = float(self.rng.uniform(-d2 + 0.9, d2 - 0.9))
        w_p0 = (w2 - 0.01, win_y, win_z - win_w / 2.0)
        w_p1 = (w2 - 0.01, win_y, win_z + win_w / 2.0)
        w_p2 = (w2 - 0.01, win_y + win_h, win_z + win_w / 2.0)
        w_p3 = (w2 - 0.01, win_y + win_h, win_z - win_w / 2.0)
        win_poly = np.array([w_p0, w_p1, w_p2, w_p3], dtype=np.float32)
        self.objects.append({
            "global_id": 3,
            "category": "window",
            "polys": [{"pts": win_poly, "color": self.win_color, "normal": np.array([-1, 0, 0], dtype=np.float32)}],
            "lines": [
                (w_p0, w_p1), (w_p1, w_p2), (w_p2, w_p3), (w_p3, w_p0),
                ((w2 - 0.01, win_y + win_h / 2.0, win_z - win_w / 2.0), (w2 - 0.01, win_y + win_h / 2.0, win_z + win_w / 2.0)),
                ((w2 - 0.01, win_y, win_z), (w2 - 0.01, win_y + win_h, win_z)),
            ],
            "color_bgr": self.win_line,
        })

        # 5. Carpet (global_id = 4)
        rw = float(self.rng.uniform(1.8, 2.3))
        rd = float(self.rng.uniform(1.3, 1.7))
        c_p0 = (tcx - rw / 2.0, 0.005, tcz - rd / 2.0)
        c_p1 = (tcx + rw / 2.0, 0.005, tcz - rd / 2.0)
        c_p2 = (tcx + rw / 2.0, 0.005, tcz + rd / 2.0)
        c_p3 = (tcx - rw / 2.0, 0.005, tcz + rd / 2.0)
        carpet_poly = np.array([c_p0, c_p1, c_p2, c_p3], dtype=np.float32)
        border = 0.12
        inner_poly = np.array([
            (c_p0[0] + border, 0.006, c_p0[2] + border),
            (c_p1[0] - border, 0.006, c_p1[2] + border),
            (c_p2[0] - border, 0.006, c_p2[2] - border),
            (c_p3[0] + border, 0.006, c_p3[2] - border),
        ], dtype=np.float32)
        inner_color = (
            int(np.clip(self.carpet_color[0] * 1.25, 0, 255)),
            int(np.clip(self.carpet_color[1] * 1.25, 0, 255)),
            int(np.clip(self.carpet_color[2] * 1.25, 0, 255)),
        )
        self.objects.append({
            "global_id": 4,
            "category": "carpet",
            "polys": [
                {"pts": carpet_poly, "color": self.carpet_color, "normal": np.array([0, 1, 0], dtype=np.float32)},
                {"pts": inner_poly, "color": inner_color, "normal": np.array([0, 1, 0], dtype=np.float32)},
            ],
            "lines": [(c_p0, c_p1), (c_p1, c_p2), (c_p2, c_p3), (c_p3, c_p0)],
            "color_bgr": self.carpet_line,
        })

        # 6. Painting (global_id = 5)
        pw = float(self.rng.uniform(0.75, 1.0))
        ph = float(self.rng.uniform(0.55, 0.75))
        py = float(self.rng.uniform(1.35, 1.65))
        px = float(self.rng.uniform(0.3, w2 - 0.7))
        p_p0 = (px - pw / 2.0, py, -d2 + 0.01)
        p_p1 = (px + pw / 2.0, py, -d2 + 0.01)
        p_p2 = (px + pw / 2.0, py + ph, -d2 + 0.01)
        p_p3 = (px - pw / 2.0, py + ph, -d2 + 0.01)
        paint_poly = np.array([p_p0, p_p1, p_p2, p_p3], dtype=np.float32)
        self.objects.append({
            "global_id": 5,
            "category": "painting",
            "polys": [{"pts": paint_poly, "color": self.paint_color, "normal": np.array([0, 0, 1], dtype=np.float32)}],
            "lines": [(p_p0, p_p1), (p_p1, p_p2), (p_p2, p_p3), (p_p3, p_p0), (p_p0, p_p2)],
            "color_bgr": self.paint_line,
        })


# ==============================================================================
# 2. Base Component Architecture
# ==============================================================================
class BaseDatasetComponent(abc.ABC):
    """Abstract Lifecycle Interface for Dataset Generator Components."""

    name: str = "base"

    def setup(self, config: Dict[str, Any]) -> None:
        """Configures component parameters prior to generation."""
        pass

    def process_view(
        self,
        view_idx: int,
        rgb: np.ndarray,
        inst_mask: np.ndarray,
        camera: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Processes and extracts modality data for a single viewpoint."""
        return {}

    def process_pair(
        self,
        view_i: int,
        view_j: int,
        view_data_i: Dict[str, Any],
        view_data_j: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Computes cross-view correspondence supervision between view pairs."""
        return {}

    def save_view(self, scene_dir: Path, view_idx: int, view_data: Dict[str, Any]) -> None:
        """Saves viewpoint-specific files to scene directory."""
        pass

    def save_scene(self, scene_dir: Path, scene_data: Dict[str, Any]) -> None:
        """Saves scene-level summary files (e.g., pairs, graphs)."""
        pass


# ==============================================================================
# 3. Component: Object & Instance Mask Component
# ==============================================================================
class ObjectMaskComponent(BaseDatasetComponent):
    """Generates semantic instance masks, bounding boxes, and cross-view object matches."""

    name = "object"

    def process_view(
        self,
        view_idx: int,
        rgb: np.ndarray,
        inst_mask: np.ndarray,
        camera: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        visible_objs = scene_context.get("visible_objects_per_view", {}).get(view_idx, [])
        return {
            "inst_mask": inst_mask,
            "objects": visible_objs,
        }

    def process_pair(
        self,
        view_i: int,
        view_j: int,
        view_data_i: Dict[str, Any],
        view_data_j: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        objs_i = view_data_i.get("objects", [])
        objs_j = view_data_j.get("objects", [])

        gid_to_idx_i = {obj["global_id"]: idx for idx, obj in enumerate(objs_i)}
        gid_to_idx_j = {obj["global_id"]: idx for idx, obj in enumerate(objs_j)}
        co_visible = set(gid_to_idx_i.keys()).intersection(set(gid_to_idx_j.keys()))

        gt_matches = [[gid_to_idx_i[g], gid_to_idx_j[g]] for g in sorted(co_visible)]
        unmatched_i = [idx for idx, obj in enumerate(objs_i) if obj["global_id"] not in co_visible]
        unmatched_j = [idx for idx, obj in enumerate(objs_j) if obj["global_id"] not in co_visible]

        return {
            "view_i": view_i,
            "view_j": view_j,
            "num_matches": len(gt_matches),
            "gt_matches": gt_matches,
            "unmatched_i": unmatched_i,
            "unmatched_j": unmatched_j,
        }

    def save_view(self, scene_dir: Path, view_idx: int, view_data: Dict[str, Any]) -> None:
        if "inst_mask" in view_data:
            cv2.imwrite(str(scene_dir / f"instance_mask_{view_idx:02d}.png"), view_data["inst_mask"])
        if "objects" in view_data:
            with open(scene_dir / f"objects_{view_idx:02d}.json", "w", encoding="utf-8") as f:
                json.dump(view_data["objects"], f, indent=2)

    def save_scene(self, scene_dir: Path, scene_data: Dict[str, Any]) -> None:
        pairs = scene_data.get("pair_data", {}).get(self.name, [])
        with open(scene_dir / "pair_matches.json", "w", encoding="utf-8") as f:
            json.dump({
                "num_pairs": len(pairs),
                "pairs": pairs,
            }, f, indent=2)


# ==============================================================================
# 4. Component: Straight Line Segment (LSD) Component (直线核心组件)
# ==============================================================================
class LSDLineComponent(BaseDatasetComponent):
    """
    Straight Line Segment Detector & Matcher Component (LSD Line Component).
    Features:
      1. OpenCV LSD line detector detects continuous straight line segments.
      2. Computes normalized geometric features: [x_c, y_c, length, sin(2θ), cos(2θ)].
      3. Associates lines with semantic objects/regions via spatial mask voting.
      4. Ground-truth cross-view line matching grounded via 3D physical edges.
      5. Exports line visual diagnostics with orientation color coding.
    """

    name = "lsd_line"

    def __init__(self, min_length: float = 16.0, refine_level: int = cv2.LSD_REFINE_STD) -> None:
        self.min_length = min_length
        self.refine_level = refine_level
        self.lsd = cv2.createLineSegmentDetector(self.refine_level)

    def setup(self, config: Dict[str, Any]) -> None:
        self.min_length = float(config.get("min_line_length", 16.0))

    def process_view(
        self,
        view_idx: int,
        rgb: np.ndarray,
        inst_mask: np.ndarray,
        camera: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        H, W = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        raw_lines, _, _, _ = self.lsd.detect(gray)

        visible_objs = scene_context.get("visible_objects_per_view", {}).get(view_idx, [])
        obj_map = {obj["inst_id"]: obj for obj in visible_objs}
        proj_lines_3d = scene_context.get("projected_lines_per_view", {}).get(view_idx, [])

        lines_meta = []
        lines_geom_feat = []

        if raw_lines is not None:
            for l_idx, l_arr in enumerate(raw_lines):
                x1, y1, x2, y2 = [float(v) for v in l_arr.flatten()]
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < self.min_length:
                    continue

                theta = float(np.arctan2(y2 - y1, x2 - x1) % np.pi)
                xc, yc = float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

                # Normalized 5D geometric feature: [xc/W, yc/H, L/max(W,H), sin2θ, cos2θ]
                feat = [
                    round(xc / W, 4),
                    round(yc / H, 4),
                    round(length / max(W, H), 4),
                    round(float(np.sin(2.0 * theta)), 4),
                    round(float(np.cos(2.0 * theta)), 4),
                ]
                lines_geom_feat.append(feat)

                # Query semantic object association along line segment
                ts = np.linspace(0.15, 0.85, 5)
                hits = []
                for t in ts:
                    sx = int(np.clip(round((1 - t) * x1 + t * x2), 0, W - 1))
                    sy = int(np.clip(round((1 - t) * y1 + t * y2), 0, H - 1))
                    hits.append(int(inst_mask[sy, sx]))

                vals, counts = np.unique(hits, return_counts=True)
                maj_inst = vals[counts.argmax()]
                if maj_inst > 0 and maj_inst in obj_map:
                    category = obj_map[maj_inst]["category"]
                    gid = obj_map[maj_inst]["global_id"]
                else:
                    category = "room_boundary"
                    gid = 0

                # Match against visible 3D wireframe edges for GT ground-truth tracking
                best_edge_id = None
                best_dist = 999.0
                for pe in proj_lines_3d:
                    px1, py1, px2, py2 = pe["coords"]
                    p_ang = np.arctan2(py2 - py1, px2 - px1) % np.pi
                    ang_diff = min(abs(theta - p_ang), np.pi - abs(theta - p_ang))
                    if ang_diff > np.radians(14.0):
                        continue

                    # Point-to-line orthogonal distance
                    num = abs((py2 - py1) * xc - (px2 - px1) * yc + px2 * py1 - py2 * px1)
                    den = np.hypot(py2 - py1, px2 - px1) + 1e-6
                    dist = num / den
                    if dist < 6.0 and dist < best_dist:
                        best_dist = dist
                        best_edge_id = pe["line_id"]

                lines_meta.append({
                    "line_id": len(lines_meta),
                    "coords": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "length": round(length, 1),
                    "angle_deg": round(float(np.degrees(theta)), 1),
                    "centroid": [round(xc, 1), round(yc, 1)],
                    "inst_id": int(maj_inst),
                    "global_id": int(gid),
                    "category": category,
                    "matched_3d_edge_id": best_edge_id,
                })

        return {
            "lines": lines_meta,
            "features": lines_geom_feat,
            "count": len(lines_meta),
        }

    def process_pair(
        self,
        view_i: int,
        view_j: int,
        view_data_i: Dict[str, Any],
        view_data_j: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        lines_i = view_data_i.get("lines", [])
        lines_j = view_data_j.get("lines", [])

        # Build edge index for view_j
        edge_map_j = {}
        for l_j in lines_j:
            eid = l_j.get("matched_3d_edge_id")
            if eid is not None:
                edge_map_j[eid] = l_j["line_id"]

        gt_line_matches = []
        for l_i in lines_i:
            eid = l_i.get("matched_3d_edge_id")
            if eid is not None and eid in edge_map_j:
                gt_line_matches.append([l_i["line_id"], edge_map_j[eid]])

        matched_i = {m[0] for m in gt_line_matches}
        matched_j = {m[1] for m in gt_line_matches}
        unmatched_i = [l["line_id"] for l in lines_i if l["line_id"] not in matched_i]
        unmatched_j = [l["line_id"] for l in lines_j if l["line_id"] not in matched_j]

        return {
            "view_i": view_i,
            "view_j": view_j,
            "num_matches": len(gt_line_matches),
            "gt_line_matches": gt_line_matches,
            "unmatched_lines_i": unmatched_i,
            "unmatched_lines_j": unmatched_j,
        }

    def save_view(self, scene_dir: Path, view_idx: int, view_data: Dict[str, Any]) -> None:
        lines = view_data.get("lines", [])
        with open(scene_dir / f"lines_lsd_{view_idx:02d}.json", "w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2)

        raw_coords = [l["coords"] for l in lines]
        with open(scene_dir / f"lines_{view_idx:02d}.json", "w", encoding="utf-8") as f:
            json.dump(raw_coords, f, indent=2)

        with open(scene_dir / f"lines_features_{view_idx:02d}.json", "w", encoding="utf-8") as f:
            json.dump(view_data.get("features", []), f, indent=2)

    def save_scene(self, scene_dir: Path, scene_data: Dict[str, Any]) -> None:
        pairs = scene_data.get("pair_data", {}).get(self.name, [])
        with open(scene_dir / "pair_line_matches.json", "w", encoding="utf-8") as f:
            json.dump({
                "num_pairs": len(pairs),
                "pairs": pairs,
            }, f, indent=2)


# ==============================================================================
# 5. Component: Wireframe 3D Analytical Edge Projection Component
# ==============================================================================
class Wireframe3DLineComponent(BaseDatasetComponent):
    """Projects 3D CAD physical wireframe edges with near-plane clipping & camera poses."""

    name = "wireframe_3d"

    def process_view(
        self,
        view_idx: int,
        rgb: np.ndarray,
        inst_mask: np.ndarray,
        camera: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        proj_lines = scene_context.get("projected_lines_per_view", {}).get(view_idx, [])
        return {"projected_lines": proj_lines}

    def save_view(self, scene_dir: Path, view_idx: int, view_data: Dict[str, Any]) -> None:
        lines = view_data.get("projected_lines", [])
        with open(scene_dir / f"lines_3d_projected_{view_idx:02d}.json", "w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2)


# ==============================================================================
# 6. Unified Scene Renderer (Coordinates Modalities for all Components)
# ==============================================================================
def render_indoor_view(
    room: UnifiedProceduralRoom,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int = 640,
    height: int = 480,
    min_pixels: int = 250,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Renders shaded RGB image, instance segmentation mask, objects, and 3D wireframe lines."""
    rgb = np.full((height, width, 3), 242, dtype=np.uint8)
    inst_mask = np.zeros((height, width), dtype=np.uint8)

    # 1. Shaded room shell
    for pl in room.planes_room:
        pts_c = [R @ pt + t for pt in pl["pts"]]
        clipped_c = clip_polygon_near_plane(pts_c, near_z=0.15)
        if len(clipped_c) >= 3:
            poly_2d = []
            for pc in clipped_c:
                px = int(K[0, 0] * pc[0] / pc[2] + K[0, 2])
                py = int(K[1, 1] * pc[1] / pc[2] + K[1, 2])
                poly_2d.append([px, py])
            shaded_color = apply_shading(pl["color"], pl["normal"], room.light_dir)
            cv2.fillPoly(rgb, np.array([poly_2d], dtype=np.int32), shaded_color)

    # Room skeleton lines
    projected_lines = []
    line_id_counter = 0

    for p1_w, p2_w in room.lines_room:
        p1_c = R @ np.array(p1_w, dtype=np.float32) + t
        p2_c = R @ np.array(p2_w, dtype=np.float32) + t
        clipped = clip_polygon_near_plane([p1_c, p2_c], near_z=0.15)
        if len(clipped) == 2:
            x1 = float(K[0, 0] * clipped[0][0] / clipped[0][2] + K[0, 2])
            y1 = float(K[1, 1] * clipped[0][1] / clipped[0][2] + K[1, 2])
            x2 = float(K[0, 0] * clipped[1][0] / clipped[1][2] + K[0, 2])
            y2 = float(K[1, 1] * clipped[1][1] / clipped[1][2] + K[1, 2])
            cv2.line(rgb, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (95, 100, 110), 2, cv2.LINE_AA)
            projected_lines.append({
                "line_id": line_id_counter,
                "global_object_id": 0,
                "category": "room_frame",
                "coords": [x1, y1, x2, y2],
            })
            line_id_counter += 1

    # 2. Render Furniture Objects
    sorted_objects = sorted(room.objects, key=lambda o: 0 if o["category"] == "carpet" else 1)
    visible_objects = []
    local_inst_id = 1

    for obj in sorted_objects:
        obj_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw filled polygons with normal shading
        for poly_info in obj["polys"]:
            pts_c = [R @ pt + t for pt in poly_info["pts"]]
            clipped_c = clip_polygon_near_plane(pts_c, near_z=0.15)
            if len(clipped_c) >= 3:
                poly_2d = []
                for pc in clipped_c:
                    px = int(round(K[0, 0] * pc[0] / pc[2] + K[0, 2]))
                    py = int(round(K[1, 1] * pc[1] / pc[2] + K[1, 2]))
                    poly_2d.append([px, py])
                shaded_col = apply_shading(poly_info["color"], poly_info["normal"], room.light_dir)
                cv2.fillPoly(rgb, np.array([poly_2d], dtype=np.int32), shaded_col)
                cv2.fillPoly(obj_mask, np.array([poly_2d], dtype=np.int32), 255)

        # Draw wireframe lines
        for p1_w, p2_w in obj["lines"]:
            p1_c = R @ np.array(p1_w, dtype=np.float32) + t
            p2_c = R @ np.array(p2_w, dtype=np.float32) + t
            clipped = clip_polygon_near_plane([p1_c, p2_c], near_z=0.15)
            if len(clipped) == 2:
                x1 = float(K[0, 0] * clipped[0][0] / clipped[0][2] + K[0, 2])
                y1 = float(K[1, 1] * clipped[0][1] / clipped[0][2] + K[1, 2])
                x2 = float(K[0, 0] * clipped[1][0] / clipped[1][2] + K[0, 2])
                y2 = float(K[1, 1] * clipped[1][1] / clipped[1][2] + K[1, 2])
                cv2.line(rgb, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), obj["color_bgr"], 3, cv2.LINE_AA)
                cv2.line(obj_mask, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, 3)
                projected_lines.append({
                    "line_id": line_id_counter,
                    "global_object_id": obj["global_id"],
                    "category": obj["category"],
                    "coords": [x1, y1, x2, y2],
                })
                line_id_counter += 1

        pix_count = int(np.count_nonzero(obj_mask))
        if pix_count >= min_pixels:
            inst_mask[obj_mask > 0] = local_inst_id
            ys, xs = np.where(obj_mask > 0)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            centroid = [float(np.mean(xs)), float(np.mean(ys))]

            visible_objects.append({
                "inst_id": local_inst_id,
                "global_id": obj["global_id"],
                "category": obj["category"],
                "bbox": [round(b, 1) for b in bbox],
                "centroid": [round(c, 1) for c in centroid],
                "pixel_count": pix_count,
            })
            local_inst_id += 1

    return rgb, inst_mask, visible_objects, projected_lines


# ==============================================================================
# 7. Dataset Generator Factory (工厂类核心)
# ==============================================================================
class DatasetGeneratorFactory:
    """
    Unified Dataset Generator Factory.
    Registers and orchestrates pluggable modal components:
      - ObjectMaskComponent
      - LSDLineComponent (The straight line component)
      - Wireframe3DLineComponent
    """

    _registry: Dict[str, Type[BaseDatasetComponent]] = {
        "object": ObjectMaskComponent,
        "lsd_line": LSDLineComponent,
        "line": LSDLineComponent,
        "wireframe_3d": Wireframe3DLineComponent,
    }

    def __init__(self, components: List[BaseDatasetComponent], config: Dict[str, Any]) -> None:
        self.components = components
        self.config = config
        for comp in self.components:
            comp.setup(config)

    @classmethod
    def register_component(cls, name: str, component_cls: Type[BaseDatasetComponent]) -> None:
        """Registers a new custom component in the factory."""
        cls._registry[name] = component_cls

    @classmethod
    def create(
        cls,
        task: str = "semantic_line",
        component_names: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "DatasetGeneratorFactory":
        """
        Factory constructor: instantiates components matching task or component list.
        Tasks:
          - 'semantic_line' (default): ObjectMaskComponent + LSDLineComponent + Wireframe3DLineComponent
          - 'object': ObjectMaskComponent
          - 'line': Wireframe3DLineComponent + LSDLineComponent
          - 'all': All registered components
        """
        cfg = config or {}
        components: List[BaseDatasetComponent] = []

        if component_names is not None:
            selected_names = list(component_names)
        elif task == "semantic_line" or task == "all":
            selected_names = ["object", "lsd_line", "wireframe_3d"]
        elif task == "object":
            selected_names = ["object"]
        elif task == "line":
            selected_names = ["lsd_line", "wireframe_3d"]
        else:
            selected_names = ["object", "lsd_line"]

        for name in selected_names:
            if name in cls._registry:
                comp_cls = cls._registry[name]
                components.append(comp_cls())
            else:
                print(f"[!] Warning: Component '{name}' not found in registry. Skipping.")

        return cls(components=components, config=cfg)

    def generate_scene(
        self,
        scene_idx: int,
        output_dir: Path,
        num_views: int = 8,
        seed: Optional[int] = None,
        random_colors: bool = True,
    ) -> None:
        """Generates a multi-view 3D scene and runs all registered components."""
        scene_dir = output_dir / f"scene_{scene_idx:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        room = UnifiedProceduralRoom(seed=seed, random_colors=random_colors)
        K = np.array([
            [380.0, 0.0, 320.0],
            [0.0, 380.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        rx = room.W * 0.38
        rz = room.D * 0.38
        target = np.array([0.0, 0.75, 0.0], dtype=np.float32)

        rendered_views: List[Dict[str, Any]] = []
        visible_objs_per_view = {}
        proj_lines_per_view = {}

        # 1. Render all views
        for v_id in range(num_views):
            angle = 2.0 * math.pi * v_id / num_views
            cam_x = float(rx * math.sin(angle))
            cam_z = float(rz * math.cos(angle))
            cam_y = float(1.35 + 0.15 * math.sin(2.0 * angle))

            eye = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
            R, t = create_look_at(eye, target)

            rgb, inst_mask, objs, lines_3d = render_indoor_view(room, K, R, t)

            # Write RGB base image
            cv2.imwrite(str(scene_dir / f"view_{v_id:02d}.png"), rgb)

            visible_objs_per_view[v_id] = objs
            proj_lines_per_view[v_id] = lines_3d

            camera_meta = {
                "R": R.tolist(),
                "t": t.tolist(),
                "eye": eye.tolist(),
                "K": K.tolist(),
            }

            rendered_views.append({
                "view_idx": v_id,
                "rgb": rgb,
                "inst_mask": inst_mask,
                "camera": camera_meta,
            })

        # Save camera metadata
        with open(scene_dir / "cameras.json", "w", encoding="utf-8") as f:
            cameras_dict = {f"view_{v['view_idx']:02d}": v["camera"] for v in rendered_views}
            json.dump({"K": K.tolist(), "views": cameras_dict}, f, indent=2)

        scene_context = {
            "scene_idx": scene_idx,
            "room": room,
            "visible_objects_per_view": visible_objs_per_view,
            "projected_lines_per_view": proj_lines_per_view,
        }

        # 2. Run View-Level Component Pipeline
        component_view_data: Dict[str, Dict[int, Dict[str, Any]]] = {c.name: {} for c in self.components}
        for v_item in rendered_views:
            v_id = v_item["view_idx"]
            for comp in self.components:
                data = comp.process_view(
                    v_id,
                    v_item["rgb"],
                    v_item["inst_mask"],
                    v_item["camera"],
                    scene_context,
                )
                comp.save_view(scene_dir, v_id, data)
                component_view_data[comp.name][v_id] = data

        # 3. Run Pair-Level Component Pipeline
        scene_pair_data: Dict[str, List[Dict[str, Any]]] = {c.name: [] for c in self.components}
        for i in range(num_views):
            for j in range(i + 1, num_views):
                for comp in self.components:
                    pair_res = comp.process_pair(
                        i,
                        j,
                        component_view_data[comp.name][i],
                        component_view_data[comp.name][j],
                        scene_context,
                    )
                    if pair_res:
                        scene_pair_data[comp.name].append(pair_res)

        scene_save_data = {
            "scene_idx": scene_idx,
            "pair_data": scene_pair_data,
        }
        for comp in self.components:
            comp.save_scene(scene_dir, scene_save_data)

    def generate(
        self,
        out_dir: Path | str,
        num_scenes: int = 50,
        views_per_scene: int = 8,
        seed: int = 42,
        random_colors: bool = True,
    ) -> None:
        """Executes full multi-scene dataset generation across all configured components."""
        p_out = Path(out_dir).resolve()
        p_out.mkdir(parents=True, exist_ok=True)

        comp_names = [c.name for c in self.components]
        print(f"=================================================================")
        print(f"      Unified Multi-Modal Dataset Generator Factory              ")
        print(f"=================================================================")
        print(f"  Active Components: {', '.join(comp_names)}")
        print(f"  Total Scenes:      {num_scenes} ({views_per_scene} views/scene)")
        print(f"  Material Mode:     {'Dimension 2 (Anti-Cheating Palettes & Shading)' if random_colors else 'Legacy Fixed'}")
        print(f"  Output Directory:  {p_out}")
        print(f"=================================================================\n")

        for s_idx in range(num_scenes):
            s_seed = seed + s_idx
            self.generate_scene(
                s_idx,
                p_out,
                num_views=views_per_scene,
                seed=s_seed,
                random_colors=random_colors,
            )
            if (s_idx + 1) % 5 == 0 or s_idx == num_scenes - 1:
                print(f"  [Generated {s_idx + 1:2d}/{num_scenes} scenes]")

        # Dataset Split Index
        num_train = max(1, int(num_scenes * 0.8))
        num_val = max(1, int(num_scenes * 0.1))
        num_test = max(1, num_scenes - num_train - num_val)

        splits = {
            "train": [f"scene_{i:04d}" for i in range(num_train)],
            "val": [f"scene_{i:04d}" for i in range(num_train, num_train + num_val)],
            "test": [f"scene_{i:04d}" for i in range(num_train + num_val, num_scenes)],
            "stats": {
                "total_scenes": num_scenes,
                "train_scenes": num_train,
                "val_scenes": num_val,
                "test_scenes": num_test,
                "components": comp_names,
                "views_per_scene": views_per_scene,
                "random_colors_dim2": random_colors,
            },
        }
        with open(p_out / "dataset_splits.json", "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

        print(f"\n[✓] Dataset successfully generated with components [{', '.join(comp_names)}] at:\n    {p_out}")


# ==============================================================================
# 8. CLI Entrypoint
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Multi-Modal Dataset Generator Factory with Pluggable Components")
    parser.add_argument("--task", type=str, default="semantic_line",
                        choices=["semantic_line", "object", "line", "all"],
                        help="Generator task preset: 'semantic_line' (objects + straight lines), 'object', 'line', or 'all'")
    parser.add_argument("--components", type=str, default=None,
                        help="Comma-separated custom components (e.g., 'object,lsd_line')")
    parser.add_argument("--num-scenes", type=int, default=10, help="Number of 3D scenes to generate (default: 10)")
    parser.add_argument("--views-per-scene", type=int, default=8, help="Number of camera views per scene (default: 8)")
    parser.add_argument("--out-dir", type=str, default="data/synthetic_semantic_lines", help="Output dataset directory")
    parser.add_argument("--min-line-len", type=float, default=16.0, help="Minimum straight line length in pixels (default: 16.0)")
    parser.add_argument("--random-colors", action="store_true", default=True, help="Enable Dimension 2 randomized material palettes & shading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    custom_comps = [c.strip() for c in args.components.split(",")] if args.components else None
    cfg = {"min_line_length": args.min_line_len}

    factory = DatasetGeneratorFactory.create(
        task=args.task,
        component_names=custom_comps,
        config=cfg,
    )

    factory.generate(
        out_dir=args.out_dir,
        num_scenes=args.num_scenes,
        views_per_scene=args.views_per_scene,
        seed=args.seed,
        random_colors=args.random_colors,
    )


__all__ = [
    "BaseDatasetComponent",
    "ObjectMaskComponent",
    "LSDLineComponent",
    "Wireframe3DLineComponent",
    "DatasetGeneratorFactory",
    "UnifiedProceduralRoom",
    "render_indoor_view",
    "create_look_at",
]


if __name__ == "__main__":
    main()
