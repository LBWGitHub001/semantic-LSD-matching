"""
Synthetic 3D Scene Modeling & Multi-View Photography Generator.

Creates parametric 3D wireframe scenes (indoor architectural room with walls, doors,
windows, furniture, paintings), captures them from virtual pinhole cameras at different
angles, and exports cleanly formatted 2D line segment matching datasets with 100% exact ground truth.

Usage:
  # Generate a single pair for inspection & parameter tuning:
  python scripts/generate_synthetic_3d_dataset.py

  # Custom view rotation and camera parameters:
  python scripts/generate_synthetic_3d_dataset.py --rotation-deg 25 --distance 3.8

  # Batch generation (when satisfied with tuning):
  python scripts/generate_synthetic_3d_dataset.py --num-pairs 50 --out-dir data/synthetic_3d_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat.matcher import LineMatcher, get_distinct_colors


# ==============================================================================
# 1. 3D Parametric Scene Modeling
# ==============================================================================
class Synthetic3DRoom:
    """
    3D Architectural Room model containing parametric geometry:
      - 12 room bounding structural edges (walls, ceiling, floor)
      - Door frame with jambs and lintel
      - Window with mullion cross-grid
      - Table with 3D top and 4 vertical legs
      - Wall-mounted photo / painting frame
      - Shaded 3D polygonal planes for realistic background rendering
    """

    def __init__(
        self,
        room_size: tuple[float, float, float] = (3.6, 2.8, 2.6),  # width (X), depth (Z), height (Y)
    ) -> None:
        self.W, self.D, self.H = room_size
        self.lines_3d: list[dict[str, Any]] = []
        self.planes_3d: list[dict[str, Any]] = []
        self._build_scene()

    def _add_line(self, p1: tuple[float, float, float], p2: tuple[float, float, float], name: str, color=(30, 30, 30)) -> None:
        line_id = len(self.lines_3d)
        self.lines_3d.append({
            "id": line_id,
            "name": name,
            "p1": np.array(p1, dtype=np.float32),
            "p2": np.array(p2, dtype=np.float32),
            "color": color,
        })

    def _add_rect_wireframe(self, corner, v1, v2, name_prefix: str, color=(40, 40, 40)) -> None:
        c = np.array(corner, dtype=np.float32)
        v1 = np.array(v1, dtype=np.float32)
        v2 = np.array(v2, dtype=np.float32)
        p0 = c
        p1 = c + v1
        p2 = c + v1 + v2
        p3 = c + v2
        self._add_line(p0, p1, f"{name_prefix}_edge0", color)
        self._add_line(p1, p2, f"{name_prefix}_edge1", color)
        self._add_line(p2, p3, f"{name_prefix}_edge2", color)
        self._add_line(p3, p0, f"{name_prefix}_edge3", color)

    def _build_scene(self) -> None:
        w2, d2, h = self.W / 2.0, self.D / 2.0, self.H

        # Floor vertices (Y = 0)
        fl_c = [
            (-w2, 0.0, -d2),  # back-left
            (w2, 0.0, -d2),   # back-right
            (w2, 0.0, d2),    # front-right
            (-w2, 0.0, d2),   # front-left
        ]
        # Ceiling vertices (Y = H)
        cl_c = [
            (-w2, h, -d2),
            (w2, h, -d2),
            (w2, h, d2),
            (-w2, h, d2),
        ]

        # 1. Room Skeleton (12 Lines)
        # Floor skirting lines
        for i in range(4):
            self._add_line(fl_c[i], fl_c[(i + 1) % 4], f"room_floor_{i}", color=(50, 40, 30))
        # Ceiling cornice lines
        for i in range(4):
            self._add_line(cl_c[i], cl_c[(i + 1) % 4], f"room_ceiling_{i}", color=(80, 80, 80))
        # 4 vertical corner columns
        for i in range(4):
            self._add_line(fl_c[i], cl_c[i], f"room_column_{i}", color=(60, 60, 60))

        # 2. Door Frame on Back Wall (Z = -d2)
        door_w, door_h = 0.9, 2.0
        door_x = -0.8
        d_bl = (door_x - door_w / 2.0, 0.0, -d2 + 0.01)
        d_br = (door_x + door_w / 2.0, 0.0, -d2 + 0.01)
        d_tl = (door_x - door_w / 2.0, door_h, -d2 + 0.01)
        d_tr = (door_x + door_w / 2.0, door_h, -d2 + 0.01)
        self._add_line(d_bl, d_tl, "door_left_jamb", color=(100, 50, 20))
        self._add_line(d_tl, d_tr, "door_lintel", color=(100, 50, 20))
        self._add_line(d_tr, d_br, "door_right_jamb", color=(100, 50, 20))

        # 3. Window on Right Wall (X = w2)
        win_w, win_h = 1.2, 1.1
        win_y = 1.0
        win_z = 0.0
        w_corner = (w2 - 0.01, win_y, win_z - win_w / 2.0)
        self._add_rect_wireframe(w_corner, (0, win_h, 0), (0, 0, win_w), "window_frame", color=(30, 90, 160))
        # Window cross mullions
        self._add_line((w2 - 0.01, win_y + win_h / 2.0, win_z - win_w / 2.0),
                       (w2 - 0.01, win_y + win_h / 2.0, win_z + win_w / 2.0), "window_mullion_h", color=(30, 90, 160))
        self._add_line((w2 - 0.01, win_y, win_z),
                       (w2 - 0.01, win_y + win_h, win_z), "window_mullion_v", color=(30, 90, 160))

        # 4. Table in Center/Left of Room
        tab_w, tab_d, tab_h = 1.1, 0.7, 0.72
        tab_cx, tab_cz = 0.3, 0.2
        t_corner = (tab_cx - tab_w / 2.0, tab_h, tab_cz - tab_d / 2.0)
        self._add_rect_wireframe(t_corner, (tab_w, 0, 0), (0, 0, tab_d), "table_top", color=(120, 70, 30))
        # 4 table legs
        legs_xz = [
            (tab_cx - tab_w / 2.0 + 0.04, tab_cz - tab_d / 2.0 + 0.04),
            (tab_cx + tab_w / 2.0 - 0.04, tab_cz - tab_d / 2.0 + 0.04),
            (tab_cx + tab_w / 2.0 - 0.04, tab_cz + tab_d / 2.0 - 0.04),
            (tab_cx - tab_w / 2.0 + 0.04, tab_cz + tab_d / 2.0 - 0.04),
        ]
        for idx, (lx, lz) in enumerate(legs_xz):
            self._add_line((lx, 0.0, lz), (lx, tab_h, lz), f"table_leg_{idx}", color=(80, 50, 20))

        # 5. Wall Painting on Back Wall (Z = -d2)
        pw, ph, py, px = 0.8, 0.6, 1.4, 0.8
        p_corner = (px - pw / 2.0, py, -d2 + 0.01)
        self._add_rect_wireframe(p_corner, (pw, 0, 0), (0, ph, 0), "painting_frame", color=(160, 30, 30))
        # Inner diagonal accent lines for visual feature richness
        self._add_line((px - pw / 2.0, py, -d2 + 0.01), (px + pw / 2.0, py + ph, -d2 + 0.01), "painting_diag1", color=(180, 100, 30))

        # Background shaded planes for rendering realistic walls
        self.planes_3d = [
            # Floor (Y=0, normal up)
            {"pts": np.array(fl_c, dtype=np.float32), "color": (210, 205, 195)},
            # Back Wall (Z=-d2)
            {"pts": np.array([fl_c[0], fl_c[1], cl_c[1], cl_c[0]], dtype=np.float32), "color": (238, 235, 230)},
            # Left Wall (X=-w2)
            {"pts": np.array([fl_c[3], fl_c[0], cl_c[0], cl_c[3]], dtype=np.float32), "color": (225, 220, 215)},
            # Right Wall (X=w2)
            {"pts": np.array([fl_c[1], fl_c[2], cl_c[2], cl_c[1]], dtype=np.float32), "color": (230, 226, 220)},
            # Ceiling (Y=h)
            {"pts": np.array(cl_c, dtype=np.float32), "color": (245, 245, 245)},
        ]


# ==============================================================================
# 2. Virtual Camera & 3D Pinhole Projection
# ==============================================================================
def create_look_at_camera(
    eye: np.ndarray,
    target: np.ndarray = np.array([0.0, 1.0, 0.0]),
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> tuple[np.ndarray, np.ndarray]:
    """Computes Camera Extrinsics [R | t] pointing towards target in standard computer vision frame."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    actual_up = np.cross(right, forward)

    # Standard CV Camera coordinates: X right, Y down, Z forward
    R = np.vstack([right, -actual_up, forward])
    t = -R @ eye
    return R.astype(np.float32), t.astype(np.float32)


def project_point_3d_to_2d(p_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
    """Projects a single 3D world coordinate to 2D pixel coordinates and camera depth."""
    p_cam = R @ p_3d + t
    z = float(p_cam[2])
    if z <= 0.05:  # Behind or too close to camera
        return np.array([-1.0, -1.0], dtype=np.float32), z

    p_proj = K @ p_cam
    x_2d = float(p_proj[0] / p_proj[2])
    y_2d = float(p_proj[1] / p_proj[2])
    return np.array([x_2d, y_2d], dtype=np.float32), z


def clip_line_frustum(
    p1_2d: np.ndarray,
    p2_2d: np.ndarray,
    z1: float,
    z2: float,
    width: int,
    height: int,
    min_length: float = 12.0,
) -> np.ndarray | None:
    """Clips 2D line segment inside image boundaries [0, width] x [0, height]."""
    if z1 <= 0.05 and z2 <= 0.05:
        return None

    # Cohen-Sutherland line clipping
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def compute_code(x, y):
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

    x1, y1 = float(p1_2d[0]), float(p1_2d[1])
    x2, y2 = float(p2_2d[0]), float(p2_2d[1])

    code1 = compute_code(x1, y1)
    code2 = compute_code(x2, y2)
    accept = False

    for _ in range(8):
        if (code1 == 0) and (code2 == 0):
            accept = True
            break
        elif (code1 & code2) != 0:
            return None
        else:
            code_out = code1 if code1 != 0 else code2
            if code_out & TOP:
                x = x1 + (x2 - x1) * (0 - y1) / max(y2 - y1, 1e-6)
                y = 0.0
            elif code_out & BOTTOM:
                x = x1 + (x2 - x1) * ((height - 1) - y1) / max(y2 - y1, 1e-6)
                y = float(height - 1)
            elif code_out & RIGHT:
                y = y1 + (y2 - y1) * ((width - 1) - x1) / max(x2 - x1, 1e-6)
                x = float(width - 1)
            elif code_out & LEFT:
                y = y1 + (y2 - y1) * (0 - x1) / max(x2 - x1, 1e-6)
                x = 0.0

            if code_out == code1:
                x1, y1 = x, y
                code1 = compute_code(x1, y1)
            else:
                x2, y2 = x, y
                code2 = compute_code(x2, y2)

    if not accept:
        return None

    length = math.hypot(x2 - x1, y2 - y1)
    if length < min_length:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ==============================================================================
# 3. Multi-View Scene Photographer & Renderer
# ==============================================================================
def render_view(
    scene: Synthetic3DRoom,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int = 640,
    height: int = 480,
    line_thickness: int = 2,
    noise_sigma: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Renders 3D room planes and projects visible 3D lines onto a 2D image."""
    canvas = np.full((height, width, 3), 240, dtype=np.uint8)

    # 1. Render 3D Background Shaded Planes
    for plane in scene.planes_3d:
        poly_pts = []
        all_valid = True
        for pt in plane["pts"]:
            p2d, z = project_point_3d_to_2d(pt, K, R, t)
            if z <= 0.05:
                all_valid = False
                break
            poly_pts.append([int(p2d[0]), int(p2d[1])])
        if all_valid and len(poly_pts) >= 3:
            pts_arr = np.array([poly_pts], dtype=np.int32)
            cv2.fillPoly(canvas, pts_arr, plane["color"])

    # 2. Project 3D Lines into 2D
    visible_lines: list[dict[str, Any]] = []
    for line_info in scene.lines_3d:
        p1_2d, z1 = project_point_3d_to_2d(line_info["p1"], K, R, t)
        p2_2d, z2 = project_point_3d_to_2d(line_info["p2"], K, R, t)

        clipped = clip_line_frustum(p1_2d, p2_2d, z1, z2, width, height)
        if clipped is not None:
            # Optional 2D noise injection (for testing robustness)
            if noise_sigma > 0:
                clipped = clipped + np.random.normal(0, noise_sigma, size=4).astype(np.float32)

            coords = [float(v) for v in clipped]
            visible_lines.append({
                "line_id_3d": line_info["id"],
                "name": line_info["name"],
                "coords": coords,
                "color": line_info["color"],
            })

            # Draw antialiased line on canvas
            pt1 = (int(round(clipped[0])), int(round(clipped[1])))
            pt2 = (int(round(clipped[2])), int(round(clipped[3])))
            cv2.line(canvas, pt1, pt2, line_info["color"], line_thickness, cv2.LINE_AA)

    return canvas, visible_lines


def generate_single_pair(
    scene: Synthetic3DRoom,
    rotation_deg: float = 22.0,
    distance: float = 3.6,
    elevation_deg: float = 22.0,
    center_y: float = 1.0,
    width: int = 640,
    height: int = 480,
    shuffle_order: bool = True,
    noise_sigma: float = 0.0,
) -> dict[str, Any]:
    """Captures two views around the 3D room with specified baseline rotation."""
    # Camera Intrinsics
    focal = 550.0
    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    target = np.array([0.0, center_y, 0.0], dtype=np.float32)

    # View 0 & View 1 camera positions on spherical orbit
    half_rot = np.deg2rad(rotation_deg / 2.0)
    elev = np.deg2rad(elevation_deg)

    y_cam = center_y + distance * math.sin(elev)
    radius_xz = distance * math.cos(elev)

    # View 0: rotated -half_rot
    eye0 = np.array([
        radius_xz * math.sin(-half_rot),
        y_cam,
        radius_xz * math.cos(-half_rot),
    ], dtype=np.float32)

    # View 1: rotated +half_rot
    eye1 = np.array([
        radius_xz * math.sin(half_rot),
        y_cam,
        radius_xz * math.cos(half_rot),
    ], dtype=np.float32)

    R0, t0 = create_look_at_camera(eye0, target)
    R1, t1 = create_look_at_camera(eye1, target)

    img0, vis_lines0 = render_view(scene, K, R0, t0, width, height, noise_sigma=noise_sigma)
    img1, vis_lines1 = render_view(scene, K, R1, t1, width, height, noise_sigma=noise_sigma)

    # Shuffle line ordering in View 1 to ensure permutation equivariance
    if shuffle_order:
        perm = np.random.permutation(len(vis_lines1))
        vis_lines1 = [vis_lines1[p] for p in perm]

    # Map 3D line ID to 2D indices
    map0 = {line["line_id_3d"]: idx for idx, line in enumerate(vis_lines0)}
    map1 = {line["line_id_3d"]: idx for idx, line in enumerate(vis_lines1)}

    gt_matches = []
    for line_id_3d, idx0 in map0.items():
        if line_id_3d in map1:
            idx1 = map1[line_id_3d]
            gt_matches.append({
                "idx0": idx0,
                "idx1": idx1,
                "line_id_3d": line_id_3d,
                "name": vis_lines0[idx0]["name"],
            })

    # Prepare numpy array coordinates (N, 4)
    lines0_arr = np.array([l["coords"] for l in vis_lines0], dtype=np.float32)
    lines1_arr = np.array([l["coords"] for l in vis_lines1], dtype=np.float32)

    return {
        "img0": img0,
        "img1": img1,
        "lines0": lines0_arr,
        "lines1": lines1_arr,
        "gt_matches": gt_matches,
        "vis_lines0": vis_lines0,
        "vis_lines1": vis_lines1,
        "camera": {
            "K": K.tolist(),
            "R0": R0.tolist(),
            "t0": t0.tolist(),
            "R1": R1.tolist(),
            "t1": t1.tolist(),
            "rotation_deg": rotation_deg,
            "baseline_meters": float(np.linalg.norm(eye1 - eye0)),
        },
    }


# ==============================================================================
# 4. Export & CLI Utilities
# ==============================================================================
def save_pair_to_disk(pair_data: dict[str, Any], out_dir: Path) -> None:
    """Exports images, line coordinates, and ground-truth matches to standard directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save RGB Images
    cv2.imwrite(str(out_dir / "image0.png"), pair_data["img0"])
    cv2.imwrite(str(out_dir / "image1.png"), pair_data["img1"])

    # 2. Save 2D Lines coordinates as clean JSON arrays: (N, 4)
    with open(out_dir / "lines0.json", "w", encoding="utf-8") as f:
        json.dump(pair_data["lines0"].tolist(), f, indent=2)

    with open(out_dir / "lines1.json", "w", encoding="utf-8") as f:
        json.dump(pair_data["lines1"].tolist(), f, indent=2)

    # 3. Save Ground Truth Matches
    with open(out_dir / "gt_matches.json", "w", encoding="utf-8") as f:
        json.dump(pair_data["gt_matches"], f, indent=2)

    # 4. Save Camera Parameters
    with open(out_dir / "camera_poses.json", "w", encoding="utf-8") as f:
        json.dump(pair_data["camera"], f, indent=2)

    # 5. Render side-by-side ground truth verification image
    matches_tuples = [(m["idx0"], m["idx1"], 1.0) for m in pair_data["gt_matches"]]
    vis = LineMatcher.draw_matches(
        pair_data["img0"],
        pair_data["img1"],
        pair_data["lines0"],
        pair_data["lines1"],
        matches_tuples,
        max_draw=len(matches_tuples),
        show_connecting_lines=True,
    )
    cv2.imwrite(str(out_dir / "vis_gt_matches.jpg"), vis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Synthetic 3D Multi-View Line Dataset")
    parser.add_argument("--num-pairs", type=int, default=1, help="Number of pairs to generate (default: 1)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic_3d_dataset" / "pair_0001",
                        help="Output directory (default: data/synthetic_3d_dataset/pair_0001)")
    parser.add_argument("--rotation-deg", type=float, default=22.0, help="Relative rotation angle in degrees (default: 22.0)")
    parser.add_argument("--distance", type=float, default=3.6, help="Camera distance from center (default: 3.6m)")
    parser.add_argument("--elevation", type=float, default=22.0, help="Camera pitch elevation angle (default: 22.0°)")
    parser.add_argument("--noise-2d", type=float, default=0.0, help="Optional 2D coordinate Gaussian noise in pixels (default: 0.0)")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle line order in View 1")
    parser.add_argument("--width", type=int, default=640, help="Image width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Image height (default: 480)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 75)
    print("      3D Parametric Architectural Scene & Multi-View Generator")
    print("=" * 75)
    print(f"[*] 3D 几何建模: 包含房间主梁柱、门框、采光窗格、桌椅家具与墙壁画框")
    scene = Synthetic3DRoom(room_size=(3.8, 3.0, 2.6))
    print(f"    - 3D 空间结构线总数: {len(scene.lines_3d)} 根")
    print(f"    - 3D 墙壁/地板多边形: {len(scene.planes_3d)} 个")

    if args.num_pairs == 1:
        print(f"\n[*] 模式: 单对高精样本生成 (供人工检查与微调)")
        print(f"    - 相对旋转角度: {args.rotation_deg}°")
        print(f"    - 相机环绕距离: {args.distance} m, 俯仰角: {args.elevation}°")
        print(f"    - 图像分辨率:   {args.width} x {args.height}")
        print(f"    - 输出保存目录: {args.out_dir}")

        pair_data = generate_single_pair(
            scene=scene,
            rotation_deg=args.rotation_deg,
            distance=args.distance,
            elevation_deg=args.elevation,
            width=args.width,
            height=args.height,
            shuffle_order=not args.no_shuffle,
            noise_sigma=args.noise_2d,
        )

        save_pair_to_disk(pair_data, args.out_dir)

        n0 = len(pair_data["lines0"])
        n1 = len(pair_data["lines1"])
        m = len(pair_data["gt_matches"])
        cam = pair_data["camera"]

        print(f"\n[+] 单对数据集生成成功！")
        print(f"    - 视角 0 (View 0) 可见线段: {n0} 根")
        print(f"    - 视角 1 (View 1) 可见线段: {n1} 根")
        print(f"    - 真实 3D 投影配对 (GT):   {m} 对")
        print(f"    - 真实相机物理基线:       {cam['baseline_meters']:.3f} 米")
        print(f"\n[+] 生成文件明细:")
        print(f"    1. {args.out_dir / 'image0.png'} (视角 0 渲染图)")
        print(f"    2. {args.out_dir / 'image1.png'} (视角 1 渲染图)")
        print(f"    3. {args.out_dir / 'lines0.json'} (View 0 线段端点坐标)")
        print(f"    4. {args.out_dir / 'lines1.json'} (View 1 打乱顺序坐标)")
        print(f"    5. {args.out_dir / 'gt_matches.json'} (精确真值对应表)")
        print(f"    6. {args.out_dir / 'camera_poses.json'} (相机内参与内外位姿)")
        print(f"    7. {args.out_dir / 'vis_gt_matches.jpg'} (彩色双目真值连线验证大图)")
        print("=" * 75)
        print("💡 提示: 您可以查看并微调参数。满意后加上 `--num-pairs 100` 即可批量一键生成！")

    else:
        # Batch generation mode
        print(f"\n[*] 批量生成模式: 即将生成 {args.num_pairs} 对多样化数据集...")
        root_dir = args.out_dir
        root_dir.mkdir(parents=True, exist_ok=True)
        index_entries = []

        for p_idx in range(1, args.num_pairs + 1):
            pair_name = f"pair_{p_idx:04d}"
            pair_dir = root_dir / pair_name

            # Randomize camera parameters across reasonable distributions
            rot = float(np.random.uniform(10.0, 35.0))
            dist = float(np.random.uniform(3.2, 4.2))
            elev = float(np.random.uniform(15.0, 30.0))

            pair_data = generate_single_pair(
                scene=scene,
                rotation_deg=rot,
                distance=dist,
                elevation_deg=elev,
                width=args.width,
                height=args.height,
                shuffle_order=True,
                noise_sigma=args.noise_2d,
            )
            save_pair_to_disk(pair_data, pair_dir)

            index_entries.append({
                "pair_id": pair_name,
                "dir": str(pair_dir.relative_to(ROOT)),
                "num_lines0": len(pair_data["lines0"]),
                "num_lines1": len(pair_data["lines1"]),
                "num_matches": len(pair_data["gt_matches"]),
                "rotation_deg": rot,
                "baseline_meters": pair_data["camera"]["baseline_meters"],
            })

            if p_idx % 10 == 0 or p_idx == args.num_pairs:
                print(f"    - 已生成 [{p_idx}/{args.num_pairs}] 对...")

        # Write index.json
        index_path = root_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({"total_pairs": len(index_entries), "pairs": index_entries}, f, indent=2)

        print(f"\n[+] 批量生成完成！共生成 {args.num_pairs} 对数据集。")
        print(f"    - 总索引文件: {index_path}")
        print("=" * 75)


if __name__ == "__main__":
    main()
