"""
Procedural 3D Room Generator, Virtual Camera Math & Software Renderer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


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
