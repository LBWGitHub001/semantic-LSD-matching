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

  # 5. Build dataset directly from Structured3D with Ground-Truth 3D Wireframe Lines:
  python generate_dataset.py --task structured3d_lines --s3d-dir Structured3D_perspective_full_00/Structured3D --out-dir data/structured3d_gt_lines --max-scenes 50
"""

from __future__ import annotations

import abc
import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from datasets.read_structured3d import (
        Camera,
        Structured3DView,
        Structured3DScene,
        STRUCTURED3D_CLASSES,
    )
    from datasets.ground_truth import (
        check_line_visibility_and_project,
        query_line_semantic,
    )
except ImportError:
    # Optional fallback if datasets module is loaded out-of-tree
    Camera = None
    Structured3DView = None
    Structured3DScene = None
    STRUCTURED3D_CLASSES = []
    check_line_visibility_and_project = None
    query_line_semantic = None


# ==============================================================================
# 0. 3D Math, Virtual Camera & Procedural Room (Imported from datasets.generators)
# ==============================================================================
from datasets.generators.procedural_room import (
    create_look_at,
    clip_polygon_near_plane,
    compute_polygon_normal,
    apply_shading,
    UnifiedProceduralRoom,
    render_indoor_view,
)


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
# 5b. Component: Real Structured3D Ground-Truth Line & Semantic Region Component
# ==============================================================================
STRUCTURED3D_TARGET_FURNITURE: list[str] = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "desk", "wardrobe", "television",
    "nightstand", "sink", "lamp", "counter",
]


def extract_structured3d_plane_masks(
    view: Any,
    H: int = 480,
    W: int = 640,
    min_pixel_area: int = 350,
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
                    "plane_id": pid,
                    "label": f"{ptype}_{pid}",
                    "category": ptype,
                    "region_type": "plane",
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "centroid_2d": [round(float(np.mean(xs)), 1), round(float(np.mean(ys)), 1)],
                    "pixel_area": area,
                    "mask": mask,
                })
    return plane_masks


def extract_structured3d_furniture_masks(
    view: Any,
    H: int = 480,
    W: int = 640,
    min_pixel_area: int = 300,
) -> list[dict[str, Any]]:
    """Extracts furniture/object region masks from semantic.png + depth.png with 3D centroids."""
    cam = view.camera
    sem_raw = view.load_semantic()
    sem = sem_raw[..., 0] if sem_raw.ndim == 3 else sem_raw
    depth = view.load_depth(in_meters=True)
    H_orig, W_orig = sem.shape

    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    R_c2w = cam.R_c2w
    t_w = cam.position / 1000.0  # Camera origin in meters

    furniture_masks = []
    for cat in STRUCTURED3D_TARGET_FURNITURE:
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
            if valid.sum() < 80:
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


def extract_structured3d_gt_lines(
    scene_anno: dict[str, Any],
    view: Any,
    target_h: int = 480,
    target_w: int = 640,
    min_length: float = 15.0,
    min_vis_ratio: float = 0.65,
    occlusion_tolerance_mm: float = 120.0,
) -> list[dict[str, Any]]:
    """
    Extracts 100% verified 2D Ground-Truth lines projected from 3D CAD wireframe annotations.
    Applies frustum clipping, near-plane clipping, and depth-buffer Z-buffer occlusion culling.
    """
    if check_line_visibility_and_project is None:
        return []

    junctions = {
        j["ID"]: np.array(j["coordinate"], dtype=np.float64)
        for j in scene_anno.get("junctions", [])
        if "ID" in j and "coordinate" in j
    }
    matrix = np.array(scene_anno.get("lineJunctionMatrix", []))
    lines_meta = scene_anno.get("lines", [])

    cam = view.camera
    depth = view.load_depth(in_meters=False)
    sem = view.load_semantic()

    scale_x = float(target_w) / float(cam.width)
    scale_y = float(target_h) / float(cam.height)

    visible_lines: list[dict[str, Any]] = []

    if matrix.ndim == 2:
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

            proj, vis = check_line_visibility_and_project(
                cam,
                depth,
                p1_w,
                p2_w,
                min_length=min_length,
                min_vis_ratio=min_vis_ratio,
                occlusion_tolerance_mm=occlusion_tolerance_mm,
            )
            if proj is None:
                continue

            (u1, v1), (u2, v2) = proj
            u1_s = float(u1) * scale_x
            v1_s = float(v1) * scale_y
            u2_s = float(u2) * scale_x
            v2_s = float(v2) * scale_y
            length_s = math.hypot(u2_s - u1_s, v2_s - v1_s)

            sem_name = query_line_semantic(sem, proj) if query_line_semantic is not None else "structure"

            visible_lines.append({
                "line_id": int(line_id),
                "coords": [round(u1_s, 1), round(v1_s, 1), round(u2_s, 1), round(v2_s, 1)],
                "length": round(length_s, 1),
                "vis_ratio": round(vis, 3),
                "semantic": sem_name,
            })

    return visible_lines


def extract_patch_gt_lines(
    regions: list[dict[str, Any]],
    visible_lines: list[dict[str, Any]],
    patch_size: int = 112,
    max_lines: int = 16,
    min_length: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clips and projects view-level GT 3D lines into each region's normalized 112x112 square patch.
    Returns:
      lines_5d: (M, max_lines, 5) float32 [c_u, c_v, l_norm, sin(2θ), cos(2θ)]
      mask: (M, max_lines) bool
      line_ids: (M, max_lines) int32
    """
    M = len(regions)
    lines_5d = np.zeros((M, max_lines, 5), dtype=np.float32)
    mask = np.zeros((M, max_lines), dtype=bool)
    line_ids = np.full((M, max_lines), -1, dtype=np.int32)

    if M == 0 or len(visible_lines) == 0:
        return lines_5d, mask, line_ids

    for k, r in enumerate(regions):
        x1, y1, x2, y2 = r["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        S = max(x2 - x1 + 1, y2 - y1 + 1)
        u_min = cx - S / 2.0
        v_min = cy - S / 2.0
        scale = float(patch_size) / float(max(S, 1e-4))

        patch_candidates = []
        for line in visible_lines:
            u1, v1, u2, v2 = line["coords"]
            p1_p = (int(round((u1 - u_min) * scale)), int(round((v1 - v_min) * scale)))
            p2_p = (int(round((u2 - u_min) * scale)), int(round((v2 - v_min) * scale)))

            ret, pt1, pt2 = cv2.clipLine((0, 0, patch_size, patch_size), p1_p, p2_p)
            if ret:
                L = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                if L >= min_length:
                    cu = float(pt1[0] + pt2[0]) / (2.0 * float(patch_size))
                    cv = float(pt1[1] + pt2[1]) / (2.0 * float(patch_size))
                    l_norm = float(L) / float(patch_size)
                    theta = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
                    sin2 = math.sin(2.0 * theta)
                    cos2 = math.cos(2.0 * theta)
                    patch_candidates.append(([cu, cv, l_norm, sin2, cos2], L, line["line_id"]))

        patch_candidates.sort(key=lambda x: x[1], reverse=True)
        for idx_l, (feat, _, lid) in enumerate(patch_candidates[:max_lines]):
            lines_5d[k, idx_l] = feat
            mask[k, idx_l] = True
            line_ids[k, idx_l] = lid

    return lines_5d, mask, line_ids


def match_structured3d_regions(
    regions0: list[dict[str, Any]],
    regions1: list[dict[str, Any]],
    max_centroid_dist: float = 0.45,
) -> tuple[list[tuple[int, int, str]], list[int], list[int]]:
    """Matches structural planes by entity_id and furniture by category + 3D centroid distance."""
    matches = []
    matched_0 = set()
    matched_1 = set()

    # 1. Match Planes by Entity ID
    plane_map_1 = {r["entity_id"]: j for j, r in enumerate(regions1) if r.get("region_type") == "plane"}
    for i, r in enumerate(regions0):
        if r.get("region_type") == "plane":
            eid = r["entity_id"]
            if eid in plane_map_1:
                j = plane_map_1[eid]
                matches.append((i, j, r["label"]))
                matched_0.add(i)
                matched_1.add(j)

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


def match_structured3d_gt_lines(
    lines0: list[dict[str, Any]],
    lines1: list[dict[str, Any]],
) -> tuple[list[tuple[int, int, int]], list[int], list[int]]:
    """Ground-truth 1-to-1 line matching via shared 3D wireframe line ID."""
    id_map_1 = {l["line_id"]: j for j, l in enumerate(lines1)}
    matches = []
    matched_0 = set()
    matched_1 = set()

    for i, l in enumerate(lines0):
        lid = l["line_id"]
        if lid in id_map_1:
            j = id_map_1[lid]
            matches.append((i, j, lid))
            matched_0.add(i)
            matched_1.add(j)

    unmatched_0 = [i for i in range(len(lines0)) if i not in matched_0]
    unmatched_1 = [j for j in range(len(lines1)) if j not in matched_1]

    return matches, unmatched_0, unmatched_1


def process_structured3d_scene(
    scene_dir_str: str,
    output_dir_str: str,
    target_h: int = 480,
    target_w: int = 640,
    min_matches: int = 3,
    max_rotation_deg: float = 85.0,
    min_area_ratio: float = 0.40,
    patch_size: int = 112,
    max_lines_per_patch: int = 16,
) -> dict[str, Any]:
    """Processes a single Structured3D scene: extracts regions, GT lines, computes correspondences."""
    scene_dir = Path(scene_dir_str)
    output_dir = Path(output_dir_str)
    scene_id = scene_dir.name

    anno_file = scene_dir / "annotation_3d.json"
    if not anno_file.is_file():
        return {"scene_id": scene_id, "pairs_created": 0}

    with open(anno_file, "r", encoding="utf-8") as f:
        scene_anno = json.load(f)

    room_dirs = sorted([d for d in (scene_dir / "2D_rendering").iterdir() if d.is_dir()])
    pairs_created = 0
    created_pair_ids = []

    for room in room_dirs:
        persp_dir = room / "perspective" / "full"
        if not persp_dir.is_dir():
            continue

        view_dirs = sorted([v for v in persp_dir.iterdir() if v.is_dir() and (v / "camera_pose.txt").is_file()])
        if len(view_dirs) < 2:
            continue

        views_dict = {}
        for vd in view_dirs:
            try:
                view_obj = Structured3DView(scene_id, room.name, vd.name, vd)
                bgr = view_obj.load_rgb()
                bgr_resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                planes = extract_structured3d_plane_masks(view_obj, target_h, target_w)
                furns = extract_structured3d_furniture_masks(view_obj, target_h, target_w)
                all_regions = planes + furns

                if len(all_regions) < 2:
                    continue

                gt_lines = extract_structured3d_gt_lines(scene_anno, view_obj, target_h, target_w)

                views_dict[vd.name] = {
                    "view_id": vd.name,
                    "bgr": bgr_resized,
                    "regions": all_regions,
                    "lines": gt_lines,
                    "cam": view_obj.camera,
                }
            except Exception:
                continue

        v_keys = list(views_dict.keys())
        for a_i in range(len(v_keys)):
            for b_i in range(a_i + 1, len(v_keys)):
                va = views_dict[v_keys[a_i]]
                vb = views_dict[v_keys[b_i]]

                cam_a, cam_b = va["cam"], vb["cam"]
                rot_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(cam_a.R_c2w.T @ cam_b.R_c2w) - 1.0) / 2.0, -1.0, 1.0))))
                base_m = float(np.linalg.norm(cam_a.position - cam_b.position) / 1000.0)

                if rot_deg > max_rotation_deg or (rot_deg < 10.0 and base_m < 0.25):
                    continue

                matches, un0, un1 = match_structured3d_regions(va["regions"], vb["regions"])
                if len(matches) < min_matches:
                    continue

                tot_area_0 = sum(r["pixel_area"] for r in va["regions"])
                tot_area_1 = sum(r["pixel_area"] for r in vb["regions"])
                matched_area_0 = sum(va["regions"][m[0]]["pixel_area"] for m in matches)
                matched_area_1 = sum(vb["regions"][m[1]]["pixel_area"] for m in matches)
                ratio_0 = matched_area_0 / tot_area_0 if tot_area_0 > 0 else 0.0
                ratio_1 = matched_area_1 / tot_area_1 if tot_area_1 > 0 else 0.0

                if min_area_ratio > 0.0 and min(ratio_0, ratio_1) < min_area_ratio:
                    continue

                # Match GT Lines
                line_matches, un_lines_0, un_lines_1 = match_structured3d_gt_lines(va["lines"], vb["lines"])

                # Extract GT lines for patches
                lines0_5d, mask0, ids0 = extract_patch_gt_lines(
                    va["regions"], va["lines"], patch_size=patch_size, max_lines=max_lines_per_patch
                )
                lines1_5d, mask1, ids1 = extract_patch_gt_lines(
                    vb["regions"], vb["lines"], patch_size=patch_size, max_lines=max_lines_per_patch
                )

                pair_id = f"{scene_id}_{room.name}_v{va['view_id']}_v{vb['view_id']}"
                pair_dir = output_dir / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(pair_dir / "image_0.png"), va["bgr"])
                cv2.imwrite(str(pair_dir / "image_1.png"), vb["bgr"])

                # Save Masks & Segmented Crops for View 0
                masks0_dir = pair_dir / "masks_0"
                crops0_dir = pair_dir / "image_0"
                masks0_dir.mkdir(exist_ok=True)
                crops0_dir.mkdir(exist_ok=True)
                meta_regs_0 = []
                for idx, r in enumerate(va["regions"]):
                    cv2.imwrite(str(masks0_dir / f"mask_{idx:02d}.png"), r["mask"])
                    x1, y1, x2, y2 = r["bbox"]
                    crop_img = va["bgr"][y1 : y2 + 1, x1 : x2 + 1]
                    crop_m = r["mask"][y1 : y2 + 1, x1 : x2 + 1]
                    bgra = np.zeros((crop_img.shape[0], crop_img.shape[1], 4), dtype=np.uint8)
                    bin_m = crop_m > 0
                    bgra[..., :3] = crop_img * bin_m[..., None]
                    bgra[..., 3] = np.where(bin_m, 255, 0).astype(np.uint8)
                    cv2.imwrite(str(crops0_dir / f"region_{idx:02d}.png"), bgra)

                    meta_regs_0.append({
                        "idx": idx,
                        "label": r["label"],
                        "category": r["category"],
                        "region_type": r["region_type"],
                        "bbox": r["bbox"],
                        "centroid_2d": r["centroid_2d"],
                        "pixel_area": r["pixel_area"],
                    })

                # Save Masks & Segmented Crops for View 1
                masks1_dir = pair_dir / "masks_1"
                crops1_dir = pair_dir / "image_1"
                masks1_dir.mkdir(exist_ok=True)
                crops1_dir.mkdir(exist_ok=True)
                meta_regs_1 = []
                for idx, r in enumerate(vb["regions"]):
                    cv2.imwrite(str(masks1_dir / f"mask_{idx:02d}.png"), r["mask"])
                    x1, y1, x2, y2 = r["bbox"]
                    crop_img = vb["bgr"][y1 : y2 + 1, x1 : x2 + 1]
                    crop_m = r["mask"][y1 : y2 + 1, x1 : x2 + 1]
                    bgra = np.zeros((crop_img.shape[0], crop_img.shape[1], 4), dtype=np.uint8)
                    bin_m = crop_m > 0
                    bgra[..., :3] = crop_img * bin_m[..., None]
                    bgra[..., 3] = np.where(bin_m, 255, 0).astype(np.uint8)
                    cv2.imwrite(str(crops1_dir / f"region_{idx:02d}.png"), bgra)

                    meta_regs_1.append({
                        "idx": idx,
                        "label": r["label"],
                        "category": r["category"],
                        "region_type": r["region_type"],
                        "bbox": r["bbox"],
                        "centroid_2d": r["centroid_2d"],
                        "pixel_area": r["pixel_area"],
                    })

                with open(pair_dir / "regions_0.json", "w", encoding="utf-8") as f:
                    json.dump(meta_regs_0, f, indent=2)
                with open(pair_dir / "regions_1.json", "w", encoding="utf-8") as f:
                    json.dump(meta_regs_1, f, indent=2)

                # Save Ground-Truth 2D Lines
                with open(pair_dir / "lines_0.json", "w", encoding="utf-8") as f:
                    json.dump(va["lines"], f, indent=2)
                with open(pair_dir / "lines_1.json", "w", encoding="utf-8") as f:
                    json.dump(vb["lines"], f, indent=2)

                # Save Precomputed Patch GT Lines NPZ (ready for instant xfeat_line_matcher training!)
                np.savez_compressed(
                    pair_dir / "patch_lines.npz",
                    lines0=lines0_5d,
                    mask0=mask0,
                    lines1=lines1_5d,
                    mask1=mask1,
                    ids0=ids0,
                    ids1=ids1,
                )

                # Save Pair Correspondence JSON
                match_records = [{"mask0_idx": m[0], "mask1_idx": m[1], "label": m[2]} for m in matches]
                line_records = [{"line0_idx": m[0], "line1_idx": m[1], "line_id": m[2]} for m in line_matches]
                corr_dict = {
                    "pair_id": pair_id,
                    "scene_id": scene_id,
                    "room_id": room.name,
                    "view0_id": va["view_id"],
                    "view1_id": vb["view_id"],
                    "rotation_deg": round(rot_deg, 2),
                    "baseline_m": round(base_m, 3),
                    "co_visible_area_ratio": round(min(ratio_0, ratio_1), 3),
                    "matches": match_records,
                    "unmatched_in_view0": un0,
                    "unmatched_in_view1": un1,
                    "line_matches": line_records,
                    "unmatched_lines_0": un_lines_0,
                    "unmatched_lines_1": un_lines_1,
                }
                with open(pair_dir / "correspondence.json", "w", encoding="utf-8") as f:
                    json.dump(corr_dict, f, indent=2)

                pairs_created += 1
                created_pair_ids.append(pair_id)

    return {"scene_id": scene_id, "pairs_created": pairs_created, "pair_ids": created_pair_ids}


class Structured3DLineRegionComponent(BaseDatasetComponent):
    """
    Pluggable Factory Component for generating Semantic Region and Ground-Truth 3D Wireframe
    Line Matching datasets directly from the raw Structured3D CAD benchmark.
    """

    name = "structured3d_lines"

    def __init__(self) -> None:
        super().__init__()
        self.s3d_dir = Path("Structured3D_perspective_full_00/Structured3D")
        self.max_scenes: Optional[int] = 50
        self.num_workers: int = os.cpu_count() or 8
        self.min_matches: int = 3
        self.min_area_ratio: float = 0.40
        self.max_rotation_deg: float = 85.0

    def setup(self, config: Dict[str, Any]) -> None:
        if "s3d_dir" in config:
            self.s3d_dir = Path(config["s3d_dir"])
        if "max_scenes" in config:
            self.max_scenes = config["max_scenes"]
        if "num_workers" in config:
            self.num_workers = config["num_workers"]
        if "min_matches" in config:
            self.min_matches = config["min_matches"]
        if "min_area_ratio" in config:
            self.min_area_ratio = config["min_area_ratio"]
        if "max_rotation" in config:
            self.max_rotation_deg = config["max_rotation"]

    def build_dataset(
        self,
        out_dir: Path | str,
        s3d_dir: Optional[Path | str] = None,
        max_scenes: Optional[int] = None,
        num_workers: Optional[int] = None,
        min_matches: int = 3,
        min_area_ratio: float = 0.40,
        max_rotation_deg: float = 85.0,
    ) -> dict[str, Any]:
        """Executes parallel dataset generation from Structured3D with Ground-Truth lines."""
        p_s3d = Path(s3d_dir or self.s3d_dir)
        p_out = Path(out_dir).resolve()
        p_out.mkdir(parents=True, exist_ok=True)

        scenes = sorted([d for d in p_s3d.iterdir() if d.is_dir() and d.name.startswith("scene_")])
        limit = max_scenes if max_scenes is not None else self.max_scenes
        if limit:
            scenes = scenes[:limit]

        workers = num_workers or self.num_workers

        print("=" * 70)
        print("   Structured3D Real-World GT Line & Region Dataset Builder (Factory)   ")
        print("=" * 70)
        print(f"  Source Directory: {p_s3d.resolve()}")
        print(f"  Output Directory: {p_out}")
        print(f"  Target Scenes:    {len(scenes)}")
        print(f"  Worker Processes: {workers}")
        print(f"  Min Matches:      {min_matches}")
        print(f"  Min Area Ratio:   {min_area_ratio * 100:.0f}%")
        print("=" * 70 + "\n")

        t0 = time.time()
        valid_scenes = []
        all_pair_ids = []
        total_pairs = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_structured3d_scene,
                    str(s),
                    str(p_out),
                    min_matches=min_matches,
                    max_rotation_deg=max_rotation_deg,
                    min_area_ratio=min_area_ratio,
                ): s.name
                for s in scenes
            }

            for idx, fut in enumerate(as_completed(futures)):
                res = fut.result()
                if res["pairs_created"] > 0:
                    valid_scenes.append(res["scene_id"])
                    total_pairs += res["pairs_created"]
                    all_pair_ids.extend(res.get("pair_ids", []))
                print(f"  [{idx+1:3d}/{len(scenes):3d}] Scene {res['scene_id']} -> {res['pairs_created']} valid pairs created.")

        valid_scenes.sort()
        all_pair_ids.sort()
        n_scenes = len(valid_scenes)
        n_pairs = len(all_pair_ids)

        if n_pairs > 0:
            n_train = int(n_pairs * 0.8)
            n_val = int(n_pairs * 0.1)
            if n_train == 0:
                train_pairs = list(all_pair_ids)
                val_pairs = list(all_pair_ids)
                test_pairs = list(all_pair_ids)
            else:
                train_pairs = all_pair_ids[:n_train]
                val_pairs = all_pair_ids[n_train : n_train + n_val]
                test_pairs = all_pair_ids[n_train + n_val :]
        else:
            train_pairs, val_pairs, test_pairs = [], [], []

        with open(p_out / "train_split.json", "w", encoding="utf-8") as f:
            json.dump(train_pairs, f, indent=2)
        with open(p_out / "val_split.json", "w", encoding="utf-8") as f:
            json.dump(val_pairs, f, indent=2)
        with open(p_out / "test_split.json", "w", encoding="utf-8") as f:
            json.dump(test_pairs, f, indent=2)

        # Also save scene-level metadata
        with open(p_out / "scenes_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "valid_scenes": valid_scenes,
                "total_scenes": n_scenes,
                "total_pairs": n_pairs,
                "train_pairs_count": len(train_pairs),
                "val_pairs_count": len(val_pairs),
                "test_pairs_count": len(test_pairs),
            }, f, indent=2)

        elapsed = time.time() - t0
        print("\n" + "=" * 70)
        print("  [✓] STRUCTURED3D GT LINE DATASET GENERATION COMPLETE!")
        print(f"  • Total Valid Scenes: {n_scenes}")
        print(f"  • Total Valid Pairs:  {n_pairs} (Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)})")
        print(f"  • Time Elapsed:       {elapsed:.1f}s ({n_pairs / max(elapsed, 1e-3):.1f} pairs/sec)")
        print(f"  • Splits Exported:    train_split.json, val_split.json, test_split.json")
        print("=" * 70)

        return {
            "total_scenes": n_scenes,
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "total_pairs": n_pairs,
            "elapsed_seconds": round(elapsed, 1),
        }


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
        "structured3d_lines": Structured3DLineRegionComponent,
        "structured3d": Structured3DLineRegionComponent,
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
          - 'structured3d_lines' / 'structured3d': Structured3DLineRegionComponent (real-world CAD GT lines)
          - 'semantic_line' (default): ObjectMaskComponent + LSDLineComponent + Wireframe3DLineComponent
          - 'object': ObjectMaskComponent
          - 'line': Wireframe3DLineComponent + LSDLineComponent
          - 'all': All registered components
        """
        cfg = config or {}
        components: List[BaseDatasetComponent] = []

        if component_names is not None:
            selected_names = list(component_names)
        elif task in ("structured3d_lines", "structured3d"):
            selected_names = ["structured3d_lines"]
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

    def generate_structured3d(
        self,
        out_dir: Path | str,
        s3d_dir: Optional[Path | str] = None,
        max_scenes: Optional[int] = None,
        num_workers: Optional[int] = None,
        min_matches: int = 3,
        min_area_ratio: float = 0.40,
        max_rotation_deg: float = 85.0,
    ) -> dict[str, Any]:
        """Builds dataset directly from raw Structured3D scenes with 100% CAD GT lines."""
        comp = None
        for c in self.components:
            if isinstance(c, Structured3DLineRegionComponent):
                comp = c
                break
        if comp is None:
            comp = Structured3DLineRegionComponent()
            comp.setup(self.config)
        return comp.build_dataset(
            out_dir=out_dir,
            s3d_dir=s3d_dir,
            max_scenes=max_scenes,
            num_workers=num_workers,
            min_matches=min_matches,
            min_area_ratio=min_area_ratio,
            max_rotation_deg=max_rotation_deg,
        )

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
                        choices=["semantic_line", "object", "line", "all", "structured3d_lines", "structured3d"],
                        help="Generator task preset: 'semantic_line' (synthetic), 'object', 'line', or 'structured3d_lines' (real CAD GT lines)")
    parser.add_argument("--components", type=str, default=None,
                        help="Comma-separated custom components (e.g., 'object,lsd_line')")
    parser.add_argument("--num-scenes", type=int, default=10, help="Number of 3D scenes to generate for synthetic tasks (default: 10)")
    parser.add_argument("--views-per-scene", type=int, default=8, help="Number of camera views per scene for synthetic tasks (default: 8)")
    parser.add_argument("--out-dir", type=str, default="data/synthetic_semantic_lines", help="Output dataset directory")
    parser.add_argument("--min-line-len", type=float, default=16.0, help="Minimum straight line length in pixels (default: 16.0)")
    parser.add_argument("--random-colors", action="store_true", default=True, help="Enable Dimension 2 randomized material palettes & shading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Arguments specific to Structured3D Real-World dataset builder
    parser.add_argument("--s3d-dir", type=str, default="Structured3D_perspective_full_00/Structured3D", help="Structured3D raw directory")
    parser.add_argument("--max-scenes", type=int, default=None, help="Max Structured3D scenes to process (default: all available)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 8, help="Parallel worker processes for Structured3D")
    parser.add_argument("--min-matches", type=int, default=3, help="Minimum matching regions per viewpoint pair")
    parser.add_argument("--min-area-ratio", type=float, default=0.40, help="Minimum mutually matched co-visible area ratio (default: 0.40)")
    parser.add_argument("--max-rotation", type=float, default=85.0, help="Maximum camera rotation in degrees (default: 85.0)")
    args = parser.parse_args()

    custom_comps = [c.strip() for c in args.components.split(",")] if args.components else None
    cfg = {
        "min_line_length": args.min_line_len,
        "s3d_dir": args.s3d_dir,
        "max_scenes": args.max_scenes,
        "num_workers": args.num_workers,
        "min_matches": args.min_matches,
        "min_area_ratio": args.min_area_ratio,
        "max_rotation": args.max_rotation,
    }

    factory = DatasetGeneratorFactory.create(
        task=args.task,
        component_names=custom_comps,
        config=cfg,
    )

    if args.task in ("structured3d_lines", "structured3d"):
        factory.generate_structured3d(
            out_dir=args.out_dir,
            s3d_dir=args.s3d_dir,
            max_scenes=args.max_scenes,
            num_workers=args.num_workers,
            min_matches=args.min_matches,
            min_area_ratio=args.min_area_ratio,
            max_rotation_deg=args.max_rotation,
        )
    else:
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
    "Structured3DLineRegionComponent",
    "DatasetGeneratorFactory",
    "UnifiedProceduralRoom",
    "render_indoor_view",
    "create_look_at",
]


if __name__ == "__main__":
    main()
