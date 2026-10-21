"""
Synthetic Multi-View Indoor Object Matching Dataset Generator.

Generates paired multi-view indoor scenes specifically designed for object-level
bipartite graph matching ("连连看"):
  1. Rendered RGB images (640x480) from indoor viewpoints.
  2. Lossless 8-bit single-channel Instance Mask images (instance_mask_XX.png) where
     pixel values indicate object instance IDs (0 = background/room, 1..K = objects).
  3. Per-view object metadata (objects_XX.json) containing:
     - inst_id: local instance ID in mask
     - category: object class name (table, door, window, carpet, painting)
     - bbox: [x_min, y_min, x_max, y_max] in pixels
     - centroid: [cx, cy]
     - pixel_count: mask area
     - global_id: permanent object ID across all views of this scene
  4. Pair ground-truth bipartite matches (pair_matches.json):
     - gt_matches: list of [idx_i, idx_j] for co-visible objects
     - unmatched_i: list of object indices in view i not visible in view j (dustbin)
     - unmatched_j: list of object indices in view j not visible in view i (dustbin)
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

# Object specifications
OBJECT_SPECS = {
    "room": {"id": 0, "name": "room_frame", "color": (95, 100, 110)},
    "table": {"id": 1, "name": "table", "color": (30, 95, 230)},       # Amber Orange
    "door": {"id": 2, "name": "door", "color": (40, 190, 50)},         # Emerald Green
    "window": {"id": 3, "name": "window", "color": (230, 150, 20)},     # Sky Blue
    "carpet": {"id": 4, "name": "carpet", "color": (190, 40, 180)},     # Deep Magenta
    "painting": {"id": 5, "name": "painting", "color": (20, 215, 245)}, # Golden Yellow
}


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
    """Sutherland-Hodgman clipping against z = near_z."""
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


class ProceduralRoom:
    """Generates 3D procedural indoor room geometry with solid polygons and wireframes."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.RandomState(seed)
        self.W = float(self.rng.uniform(3.4, 4.4))  # Room width
        self.D = float(self.rng.uniform(3.4, 4.4))  # Room depth
        self.H = float(self.rng.uniform(2.6, 2.9))  # Room height

        # Colors
        self.floor_color = (int(self.rng.uniform(180, 210)), int(self.rng.uniform(180, 210)), int(self.rng.uniform(180, 210)))
        self.wall_color = (int(self.rng.uniform(225, 240)), int(self.rng.uniform(225, 240)), int(self.rng.uniform(225, 240)))

        self.objects: list[dict[str, Any]] = []
        self._build_geometry()

    def _build_geometry(self) -> None:
        w2, d2, h = self.W / 2.0, self.D / 2.0, self.H

        # 1. Room Shell (global_id = 0, background)
        fl_c = [(-w2, 0.0, -d2), (w2, 0.0, -d2), (w2, 0.0, d2), (-w2, 0.0, d2)]
        cl_c = [(-w2, h, -d2), (w2, h, -d2), (w2, h, d2), (-w2, h, d2)]

        self.planes_room = [
            {"pts": np.array(fl_c, dtype=np.float32), "color": self.floor_color},
            {"pts": np.array([fl_c[0], fl_c[1], cl_c[1], cl_c[0]], dtype=np.float32), "color": self.wall_color},
            {"pts": np.array([fl_c[3], fl_c[0], cl_c[0], cl_c[3]], dtype=np.float32), "color": (int(self.wall_color[0]*0.95), int(self.wall_color[1]*0.95), int(self.wall_color[2]*0.95))},
            {"pts": np.array([fl_c[1], fl_c[2], cl_c[2], cl_c[1]], dtype=np.float32), "color": (int(self.wall_color[0]*0.97), int(self.wall_color[1]*0.97), int(self.wall_color[2]*0.97))},
            {"pts": np.array(cl_c, dtype=np.float32), "color": (245, 245, 245)},
        ]

        # Room skeleton lines
        self.lines_room = []
        for i in range(4):
            self.lines_room.append((fl_c[i], fl_c[(i + 1) % 4]))
            self.lines_room.append((cl_c[i], cl_c[(i + 1) % 4]))
            self.lines_room.append((fl_c[i], cl_c[i]))

        # 2. Table (global_id = 1)
        tw = float(self.rng.uniform(1.1, 1.4))
        td = float(self.rng.uniform(0.7, 0.9))
        th = float(self.rng.uniform(0.72, 0.78))
        tcx = float(self.rng.uniform(-0.15, 0.45))
        tcz = float(self.rng.uniform(-0.25, 0.35))
        c0 = (tcx - tw / 2.0, th, tcz - td / 2.0)
        c1 = (tcx + tw / 2.0, th, tcz - td / 2.0)
        c2 = (tcx + tw / 2.0, th, tcz + td / 2.0)
        c3 = (tcx - tw / 2.0, th, tcz + td / 2.0)
        table_top_poly = np.array([c0, c1, c2, c3], dtype=np.float32)
        table_lines = [
            (c0, c1), (c1, c2), (c2, c3), (c3, c0),
            (c0, (c0[0], 0.0, c0[2])),
            (c1, (c1[0], 0.0, c1[2])),
            (c2, (c2[0], 0.0, c2[2])),
            (c3, (c3[0], 0.0, c3[2])),
        ]
        self.objects.append({
            "global_id": 1,
            "category": "table",
            "polys": [table_top_poly],
            "lines": table_lines,
            "color_bgr": OBJECT_SPECS["table"]["color"],
            "fill_color": (35, 110, 245),
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
        door_lines = [(d_p0, d_p3), (d_p3, d_p2), (d_p2, d_p1)]
        self.objects.append({
            "global_id": 2,
            "category": "door",
            "polys": [door_poly],
            "lines": door_lines,
            "color_bgr": OBJECT_SPECS["door"]["color"],
            "fill_color": (50, 205, 70),
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
        win_lines = [
            (w_p0, w_p1), (w_p1, w_p2), (w_p2, w_p3), (w_p3, w_p0),
            ((w2 - 0.01, win_y + win_h / 2.0, win_z - win_w / 2.0), (w2 - 0.01, win_y + win_h / 2.0, win_z + win_w / 2.0)),
            ((w2 - 0.01, win_y, win_z), (w2 - 0.01, win_y + win_h, win_z)),
        ]
        self.objects.append({
            "global_id": 3,
            "category": "window",
            "polys": [win_poly],
            "lines": win_lines,
            "color_bgr": OBJECT_SPECS["window"]["color"],
            "fill_color": (245, 175, 45),
        })

        # 5. Carpet (global_id = 4) on Floor (Y = 0.005)
        rw = float(self.rng.uniform(1.8, 2.3))
        rd = float(self.rng.uniform(1.3, 1.7))
        c_p0 = (tcx - rw / 2.0, 0.005, tcz - rd / 2.0)
        c_p1 = (tcx + rw / 2.0, 0.005, tcz - rd / 2.0)
        c_p2 = (tcx + rw / 2.0, 0.005, tcz + rd / 2.0)
        c_p3 = (tcx - rw / 2.0, 0.005, tcz + rd / 2.0)
        carpet_poly = np.array([c_p0, c_p1, c_p2, c_p3], dtype=np.float32)
        carpet_lines = [(c_p0, c_p1), (c_p1, c_p2), (c_p2, c_p3), (c_p3, c_p0)]
        self.objects.append({
            "global_id": 4,
            "category": "carpet",
            "polys": [carpet_poly],
            "lines": carpet_lines,
            "color_bgr": OBJECT_SPECS["carpet"]["color"],
            "fill_color": (210, 60, 200),
        })

        # 6. Painting (global_id = 5) on Back Wall (Z = -d2)
        pw = float(self.rng.uniform(0.75, 1.0))
        ph = float(self.rng.uniform(0.55, 0.75))
        py = float(self.rng.uniform(1.35, 1.65))
        px = float(self.rng.uniform(0.3, w2 - 0.7))
        p_p0 = (px - pw / 2.0, py, -d2 + 0.01)
        p_p1 = (px + pw / 2.0, py, -d2 + 0.01)
        p_p2 = (px + pw / 2.0, py + ph, -d2 + 0.01)
        p_p3 = (px - pw / 2.0, py + ph, -d2 + 0.01)
        paint_poly = np.array([p_p0, p_p1, p_p2, p_p3], dtype=np.float32)
        paint_lines = [
            (p_p0, p_p1), (p_p1, p_p2), (p_p2, p_p3), (p_p3, p_p0),
            (p_p0, p_p2),
        ]
        self.objects.append({
            "global_id": 5,
            "category": "painting",
            "polys": [paint_poly],
            "lines": paint_lines,
            "color_bgr": OBJECT_SPECS["painting"]["color"],
            "fill_color": (30, 225, 250),
        })


def render_view_with_instance_masks(
    room: ProceduralRoom,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int = 640,
    height: int = 480,
    min_pixels: int = 300,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Renders RGB image and single-channel 8-bit instance mask.
    Returns:
      rgb: (H, W, 3) uint8 image
      inst_mask: (H, W) uint8 mask (0 = background, k = instance ID)
      objects_meta: list of visible objects with bounding boxes and global_id
    """
    rgb = np.full((height, width, 3), 242, dtype=np.uint8)
    inst_mask = np.zeros((height, width), dtype=np.uint8)

    # 1. Render room shading
    for pl in room.planes_room:
        pts_c = [R @ pt + t for pt in pl["pts"]]
        clipped_c = clip_polygon_near_plane(pts_c, near_z=0.15)
        if len(clipped_c) >= 3:
            poly_2d = []
            for pc in clipped_c:
                px = int(K[0, 0] * pc[0] / pc[2] + K[0, 2])
                py = int(K[1, 1] * pc[1] / pc[2] + K[1, 2])
                poly_2d.append([px, py])
            cv2.fillPoly(rgb, np.array([poly_2d], dtype=np.int32), pl["color"])

    # Render room lines
    for p1_w, p2_w in room.lines_room:
        p1_c = R @ np.array(p1_w, dtype=np.float32) + t
        p2_c = R @ np.array(p2_w, dtype=np.float32) + t
        clipped = clip_polygon_near_plane([p1_c, p2_c], near_z=0.15)
        if len(clipped) == 2:
            x1 = int(round(K[0, 0] * clipped[0][0] / clipped[0][2] + K[0, 2]))
            y1 = int(round(K[1, 1] * clipped[0][1] / clipped[0][2] + K[1, 2]))
            x2 = int(round(K[0, 0] * clipped[1][0] / clipped[1][2] + K[0, 2]))
            y2 = int(round(K[1, 1] * clipped[1][1] / clipped[1][2] + K[1, 2]))
            cv2.line(rgb, (x1, y1), (x2, y2), OBJECT_SPECS["room"]["color"], 2, cv2.LINE_AA)

    # 2. Render each object onto RGB and Instance Mask
    # Order: carpet first (floor level), then other furniture
    sorted_objects = sorted(room.objects, key=lambda o: 0 if o["category"] == "carpet" else 1)

    visible_objects = []
    local_inst_id = 1

    for obj in sorted_objects:
        obj_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw filled polygons
        for poly in obj["polys"]:
            pts_c = [R @ pt + t for pt in poly]
            clipped_c = clip_polygon_near_plane(pts_c, near_z=0.15)
            if len(clipped_c) >= 3:
                poly_2d = []
                for pc in clipped_c:
                    px = int(round(K[0, 0] * pc[0] / pc[2] + K[0, 2]))
                    py = int(round(K[1, 1] * pc[1] / pc[2] + K[1, 2]))
                    poly_2d.append([px, py])
                cv2.fillPoly(rgb, np.array([poly_2d], dtype=np.int32), obj["fill_color"])
                cv2.fillPoly(obj_mask, np.array([poly_2d], dtype=np.int32), 255)

        # Draw lines/wireframes
        for p1_w, p2_w in obj["lines"]:
            p1_c = R @ np.array(p1_w, dtype=np.float32) + t
            p2_c = R @ np.array(p2_w, dtype=np.float32) + t
            clipped = clip_polygon_near_plane([p1_c, p2_c], near_z=0.15)
            if len(clipped) == 2:
                x1 = int(round(K[0, 0] * clipped[0][0] / clipped[0][2] + K[0, 2]))
                y1 = int(round(K[1, 1] * clipped[0][1] / clipped[0][2] + K[1, 2]))
                x2 = int(round(K[0, 0] * clipped[1][0] / clipped[1][2] + K[0, 2]))
                y2 = int(round(K[1, 1] * clipped[1][1] / clipped[1][2] + K[1, 2]))
                cv2.line(rgb, (x1, y1), (x2, y2), obj["color_bgr"], 3, cv2.LINE_AA)
                cv2.line(obj_mask, (x1, y1), (x2, y2), 255, 3)

        # Check visibility
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

    return rgb, inst_mask, visible_objects


def generate_scene(
    scene_idx: int,
    output_dir: Path,
    num_views: int = 8,
    seed: int | None = None,
) -> None:
    """Generates all views and paired bipartite ground-truth matches for one scene."""
    scene_dir = output_dir / f"scene_{scene_idx:04d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    room = ProceduralRoom(seed=seed)
    K = np.array([
        [380.0, 0.0, 320.0],
        [0.0, 380.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # Place cameras inside the room
    rx = room.W * 0.38
    rz = room.D * 0.38
    target = np.array([0.0, 0.75, 0.0], dtype=np.float32)

    views_meta = []
    views_objects = []

    for v_id in range(num_views):
        angle = 2.0 * math.pi * v_id / num_views
        cam_x = float(rx * math.sin(angle))
        cam_z = float(rz * math.cos(angle))
        cam_y = float(1.35 + 0.15 * math.sin(2.0 * angle))

        eye = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        R, t = create_look_at(eye, target)

        rgb, inst_mask, objs = render_view_with_instance_masks(room, K, R, t)

        # Save RGB image and instance mask
        cv2.imwrite(str(scene_dir / f"view_{v_id:02d}.png"), rgb)
        cv2.imwrite(str(scene_dir / f"instance_mask_{v_id:02d}.png"), inst_mask)

        # Save object metadata
        with open(scene_dir / f"objects_{v_id:02d}.json", "w", encoding="utf-8") as f:
            json.dump(objs, f, indent=2)

        views_meta.append({"view_id": v_id, "num_objects": len(objs), "cam_pos": [round(c, 2) for c in eye.tolist()]})
        views_objects.append(objs)

    # 3. Compute Bipartite Ground-Truth Matches for all pairs (i, j)
    pair_matches = []
    for i in range(num_views):
        for j in range(i + 1, num_views):
            objs_i = views_objects[i]
            objs_j = views_objects[j]

            # Map global_id to local index
            gid_to_idx_i = {obj["global_id"]: idx for idx, obj in enumerate(objs_i)}
            gid_to_idx_j = {obj["global_id"]: idx for idx, obj in enumerate(objs_j)}

            co_visible_gids = set(gid_to_idx_i.keys()).intersection(set(gid_to_idx_j.keys()))

            gt_matches = []
            for gid in sorted(co_visible_gids):
                gt_matches.append([gid_to_idx_i[gid], gid_to_idx_j[gid]])

            unmatched_i = [idx for idx, obj in enumerate(objs_i) if obj["global_id"] not in co_visible_gids]
            unmatched_j = [idx for idx, obj in enumerate(objs_j) if obj["global_id"] not in co_visible_gids]

            pair_matches.append({
                "view_i": i,
                "view_j": j,
                "num_matches": len(gt_matches),
                "gt_matches": gt_matches,
                "unmatched_i": unmatched_i,
                "unmatched_j": unmatched_j,
            })

    with open(scene_dir / "pair_matches.json", "w", encoding="utf-8") as f:
        json.dump({
            "scene_idx": scene_idx,
            "num_views": num_views,
            "pairs": pair_matches,
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic object matching dataset")
    parser.add_argument("--num-scenes", type=int, default=50, help="Number of 3D scenes to generate")
    parser.add_argument("--views-per-scene", type=int, default=8, help="Number of camera views per scene")
    parser.add_argument("--out-dir", type=str, default="data/synthetic_object_matching_scenes", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_scenes} object matching scenes into {out_dir}...")
    for s_idx in range(args.num_scenes):
        s_seed = args.seed + s_idx
        generate_scene(s_idx, out_dir, num_views=args.views_per_scene, seed=s_seed)
        if (s_idx + 1) % 10 == 0 or s_idx == args.num_scenes - 1:
            print(f"  [Generated {s_idx + 1}/{args.num_scenes} scenes]")

    # Create dataset split index (80% train / 10% val / 10% test)
    num_train = int(args.num_scenes * 0.8)
    num_val = int(args.num_scenes * 0.1)
    num_test = args.num_scenes - num_train - num_val

    splits = {
        "train": [f"scene_{i:04d}" for i in range(num_train)],
        "val": [f"scene_{i:04d}" for i in range(num_train, num_train + num_val)],
        "test": [f"scene_{i:04d}" for i in range(num_train + num_val, args.num_scenes)],
        "stats": {
            "total_scenes": args.num_scenes,
            "train_scenes": num_train,
            "val_scenes": num_val,
            "test_scenes": num_test,
            "views_per_scene": args.views_per_scene,
            "pairs_per_scene": args.views_per_scene * (args.views_per_scene - 1) // 2,
        }
    }
    with open(out_dir / "dataset_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Dataset complete! Split: {num_train} train / {num_val} val / {num_test} test.")


if __name__ == "__main__":
    main()
