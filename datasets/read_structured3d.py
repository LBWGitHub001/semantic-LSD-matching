#!/usr/bin/env python3
"""
Structured3D Dataset Reader & Inspector.

Provides clean object-oriented APIs for loading Structured3D perspective viewpoints,
parsing camera poses and intrinsic matrices, reading RGB, depth, normal, semantic maps,
and projecting 3D wireframe lines / 2D layouts.

Usage:
  python tools/read_structured3d.py
  python tools/read_structured3d.py --stats-only
  python tools/read_structured3d.py --scene scene_00005 --room 827475 --view 0
  python tools/read_structured3d.py --pair
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Structured3D standard 40 NYUv2 semantic categories (+ void)
STRUCTURED3D_CLASSES: list[str] = [
    "void",
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "fridge",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "nightstand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
]


def generate_semantic_colormap(num_classes: int = 41) -> np.ndarray:
    """Generate a distinct, reproducible RGB colormap for semantic classes."""
    np.random.seed(42)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # void is black
    palette[1] = [174, 199, 232]  # wall
    palette[2] = [152, 223, 138]  # floor
    palette[8] = [255, 127, 14]  # door
    palette[9] = [31, 119, 180]  # window
    palette[22] = [214, 39, 40]  # ceiling

    for i in range(1, num_classes):
        if np.all(palette[i] == 0):
            # Generate distinct HSL-like hues
            hue = int(255 * (i / num_classes))
            bgr = cv2.applyColorMap(
                np.array([[hue]], dtype=np.uint8), cv2.COLORMAP_HSV
            )[0, 0]
            palette[i] = [bgr[2], bgr[1], bgr[0]]
    return palette


@dataclass
class Camera:
    """
    Structured3D perspective pinhole camera model.

    Coordinates convention:
      - World frame: Right-handed, Z is upward (millimeters).
      - Camera frame: OpenCV standard (X: Right, Y: Down, Z: Forward/Look).
    """

    position: np.ndarray  # (3,) camera origin T in world coordinates (mm)
    look: np.ndarray  # (3,) camera viewing unit vector (world coordinates)
    up: np.ndarray  # (3,) camera upward unit vector (world coordinates)
    right: np.ndarray  # (3,) camera rightward unit vector (world coordinates)
    xfov: float  # horizontal half-FOV in radians
    yfov: float  # vertical half-FOV in radians
    width: int = 1280
    height: int = 720
    scale: float = 1.0

    fx: float = field(init=False)
    fy: float = field(init=False)
    cx: float = field(init=False)
    cy: float = field(init=False)
    K: np.ndarray = field(init=False)  # 3x3 intrinsics
    K_inv: np.ndarray = field(init=False)
    R_c2w: np.ndarray = field(init=False)  # 3x3 rotation: camera to world
    R_w2c: np.ndarray = field(init=False)  # 3x3 rotation: world to camera
    c2w: np.ndarray = field(init=False)  # 4x4 camera to world
    w2c: np.ndarray = field(init=False)  # 4x4 world to camera

    def __post_init__(self) -> None:
        self.fx = float((self.width / 2.0) / np.tan(self.xfov))
        self.fy = float((self.height / 2.0) / np.tan(self.yfov))
        self.cx = float(self.width / 2.0)
        self.cy = float(self.height / 2.0)

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.K_inv = np.linalg.inv(self.K)

        # In camera coordinates: X is right, Y is down (-up), Z is look
        x_axis = self.right
        y_axis = -self.up
        z_axis = self.look

        self.R_c2w = np.column_stack([x_axis, y_axis, z_axis])
        self.R_w2c = self.R_c2w.T

        t_w2c = -self.R_w2c @ self.position

        self.c2w = np.eye(4, dtype=np.float64)
        self.c2w[:3, :3] = self.R_c2w
        self.c2w[:3, 3] = self.position

        self.w2c = np.eye(4, dtype=np.float64)
        self.w2c[:3, :3] = self.R_w2c
        self.w2c[:3, 3] = t_w2c

    @classmethod
    def from_file(
        cls, pose_path: str | Path, width: int = 1280, height: int = 720
    ) -> Camera:
        """Parse camera_pose.txt containing 12 parameters."""
        with open(pose_path, "r", encoding="utf-8") as handle:
            vals = [float(x) for x in handle.read().strip().split()]
        if len(vals) != 12:
            raise ValueError(
                f"Expected 12 parameters in {pose_path}, got {len(vals)}"
            )

        position = np.array(vals[0:3], dtype=np.float64)
        look = np.array(vals[3:6], dtype=np.float64)
        up = np.array(vals[6:9], dtype=np.float64)
        xfov = vals[9]
        yfov = vals[10]
        scale = vals[11]

        look = look / np.linalg.norm(look)
        up = up / np.linalg.norm(up)
        right = np.cross(look, up)
        right = right / np.linalg.norm(right)

        return cls(
            position=position,
            look=look,
            up=up,
            right=right,
            xfov=xfov,
            yfov=yfov,
            width=width,
            height=height,
            scale=scale,
        )

    def world_to_cam(self, pts_3d_world: np.ndarray) -> np.ndarray:
        """Transform 3D world points (N, 3) to camera frame (N, 3)."""
        pts = np.asarray(pts_3d_world, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts.reshape(1, 3)
        pts_cam = (pts - self.position) @ self.R_w2c.T
        return pts_cam[0] if single else pts_cam

    def cam_to_world(self, pts_3d_cam: np.ndarray) -> np.ndarray:
        """Transform 3D camera points (N, 3) to world frame (N, 3)."""
        pts = np.asarray(pts_3d_cam, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts.reshape(1, 3)
        pts_world = (pts @ self.R_c2w.T) + self.position
        return pts_world[0] if single else pts_world

    def project(
        self, pts_3d_world: np.ndarray, min_depth: float = 50.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D world points to 2D image pixels.

        Returns:
            uv: (N, 2) pixel coordinates
            valid: (N,) boolean mask (in front of camera and inside image bounds)
            depths: (N,) camera Z depths in mm
        """
        pts_cam = self.world_to_cam(pts_3d_world)
        if pts_cam.ndim == 1:
            pts_cam = pts_cam.reshape(1, 3)

        z = pts_cam[:, 2]
        depth_valid = z >= min_depth

        u = self.fx * (pts_cam[:, 0] / np.maximum(z, 1e-6)) + self.cx
        v = self.fy * (pts_cam[:, 1] / np.maximum(z, 1e-6)) + self.cy
        uv = np.column_stack([u, v])

        in_bounds = (
            (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height) & depth_valid
        )
        return uv, in_bounds, z

    def project_line_segment(
        self,
        p1_3d_world: np.ndarray,
        p2_3d_world: np.ndarray,
        min_depth: float = 50.0,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """
        Project a 3D line segment into the image plane with near-plane and frustum clipping.

        Returns ((u1, v1), (u2, v2)) in pixel coordinates, or None if outside field of view.
        """
        p1_cam = self.world_to_cam(p1_3d_world)
        p2_cam = self.world_to_cam(p2_3d_world)
        z1, z2 = p1_cam[2], p2_cam[2]

        if z1 < min_depth and z2 < min_depth:
            return None

        # Near plane clipping
        if z1 < min_depth:
            t = (min_depth - z1) / (z2 - z1 + 1e-8)
            p1_cam = p1_cam + t * (p2_cam - p1_cam)
        elif z2 < min_depth:
            t = (min_depth - z2) / (z1 - z2 + 1e-8)
            p2_cam = p2_cam + t * (p1_cam - p2_cam)

        u1 = int(round(self.fx * (p1_cam[0] / p1_cam[2]) + self.cx))
        v1 = int(round(self.fy * (p1_cam[1] / p1_cam[2]) + self.cy))
        u2 = int(round(self.fx * (p2_cam[0] / p2_cam[2]) + self.cx))
        v2 = int(round(self.fy * (p2_cam[1] / p2_cam[2]) + self.cy))

        clipped, pt1, pt2 = cv2.clipLine(
            (0, 0, self.width, self.height), (u1, v1), (u2, v2)
        )
        if not clipped:
            return None
        return pt1, pt2

    def relative_pose_to(
        self, other: Camera
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute relative transformation from this camera (1) to other camera (2).

        Returns:
            R_rel: 3x3 relative rotation (R2 @ R1.T)
            t_rel: 3x1 relative translation in camera 2 frame
            rot_deg: rotation angle error in degrees
            baseline_m: camera baseline distance in meters
        """
        R_rel = other.R_w2c @ self.R_w2c.T
        t_rel = other.w2c[:3, 3] - R_rel @ self.w2c[:3, 3]
        baseline_m = float(np.linalg.norm(self.position - other.position) / 1000.0)

        # Angle from rotation matrix trace
        tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        rot_deg = float(np.degrees(np.arccos(tr)))
        return R_rel, t_rel, rot_deg, baseline_m


class Structured3DView:
    """Represents a single camera viewpoint with lazy-loading helpers."""

    def __init__(
        self, scene_id: str, room_id: str, view_id: str, view_dir: Path
    ) -> None:
        self.scene_id = scene_id
        self.room_id = room_id
        self.view_id = view_id
        self.view_dir = view_dir
        self.camera_pose_path = view_dir / "camera_pose.txt"
        self._camera: Camera | None = None

    @property
    def camera(self) -> Camera:
        if self._camera is None:
            self._camera = Camera.from_file(self.camera_pose_path)
        return self._camera

    def load_rgb(self, rawlight: bool = True, bgr: bool = True) -> np.ndarray:
        """Load RGB image. Returns (H, W, 3) uint8."""
        fname = "rgb_rawlight.png" if rawlight else "albedo.png"
        path = self.view_dir / fname
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing RGB image: {path}")
        if not bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_depth(self, in_meters: bool = False) -> np.ndarray:
        """
        Load 16-bit depth map.

        Args:
            in_meters: If True, returns float32 in meters; else uint16 in mm.
        """
        path = self.view_dir / "depth.png"
        depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise FileNotFoundError(f"Missing depth map: {path}")
        if in_meters:
            return depth.astype(np.float32) / 1000.0
        return depth

    def load_normal(self, decode_vectors: bool = False) -> np.ndarray:
        """
        Load surface normal map.

        Args:
            decode_vectors: If True, decodes to float32 (H, W, 3) unit normal vectors in [-1, 1].
        """
        path = self.view_dir / "normal.png"
        normal = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if normal is None:
            raise FileNotFoundError(f"Missing normal map: {path}")
        if decode_vectors:
            # OpenCV BGR -> RGB [-1, 1]
            rgb = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB).astype(np.float32)
            vec = (rgb / 127.5) - 1.0
            norm = np.linalg.norm(vec, axis=-1, keepdims=True)
            norm = np.maximum(norm, 1e-6)
            return vec / norm
        return normal

    def load_semantic(self) -> np.ndarray:
        """Load 8-bit semantic mask. Returns (H, W) uint8 with class IDs (1..40)."""
        path = self.view_dir / "semantic.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing semantic map: {path}")
        from PIL import Image
        return np.array(Image.open(path), dtype=np.uint8)

    def load_layout(self) -> dict[str, Any]:
        """Load 2D layout annotations."""
        path = self.view_dir / "layout.json"
        if not path.is_file():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def colorized_depth(self) -> np.ndarray:
        """Return 3-channel BGR visualization of depth map."""
        depth = self.load_depth(in_meters=False)
        valid = depth > 0
        norm = np.zeros_like(depth, dtype=np.uint8)
        if np.any(valid):
            d_min, d_max = np.percentile(depth[valid], [2, 98])
            clipped = np.clip(depth, d_min, d_max)
            norm[valid] = (
                (clipped[valid] - d_min) / max(d_max - d_min, 1) * 255
            ).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

    def colorized_semantic(self) -> np.ndarray:
        """Return 3-channel BGR visualization of semantic mask."""
        mask = self.load_semantic()
        palette = generate_semantic_colormap(len(STRUCTURED3D_CLASSES))
        color = palette[np.clip(mask, 0, len(STRUCTURED3D_CLASSES) - 1)]
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    def __repr__(self) -> str:
        return f"Structured3DView(scene={self.scene_id}, room={self.room_id}, view={self.view_id})"


class Structured3DScene:
    """Represents a scene with 3D structural annotations and constituent viewpoints."""

    def __init__(self, scene_id: str, scene_dir: Path) -> None:
        self.scene_id = scene_id
        self.scene_dir = scene_dir
        self.annotation_3d_path = scene_dir / "annotation_3d.json"
        self._annotation_3d: dict[str, Any] | None = None
        self._views: list[Structured3DView] | None = None

    def load_annotation_3d(self) -> dict[str, Any]:
        """Load scene-level annotation_3d.json."""
        if self._annotation_3d is None:
            if not self.annotation_3d_path.is_file():
                raise FileNotFoundError(
                    f"Missing annotation_3d.json in {self.scene_dir}"
                )
            with open(self.annotation_3d_path, "r", encoding="utf-8") as handle:
                self._annotation_3d = json.load(handle)
        return self._annotation_3d

    def get_3d_lines(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Extract all 3D line segment endpoints from lineJunctionMatrix.

        Returns list of (p1, p2) in world coordinates (mm).
        """
        anno = self.load_annotation_3d()
        junctions = anno.get("junctions", [])
        matrix = np.array(anno.get("lineJunctionMatrix", []))
        coords = {
            j["ID"]: np.array(j["coordinate"], dtype=np.float64)
            for j in junctions
            if "ID" in j and "coordinate" in j
        }

        segments: list[tuple[np.ndarray, np.ndarray]] = []
        if matrix.ndim == 2:
            for row in matrix:
                idx = np.where(row == 1)[0]
                if len(idx) == 2 and idx[0] in coords and idx[1] in coords:
                    segments.append((coords[idx[0]], coords[idx[1]]))
        return segments

    def get_views(self) -> list[Structured3DView]:
        """Scan and return all perspective viewpoints in this scene."""
        if self._views is not None:
            return self._views
        views: list[Structured3DView] = []
        pattern = self.scene_dir / "2D_rendering" / "*" / "perspective" / "full" / "*"
        for vpath in sorted(glob.glob(str(pattern))):
            p = Path(vpath)
            if p.is_dir() and (p / "camera_pose.txt").is_file():
                parts = p.parts
                room_id = parts[-4]
                view_id = parts[-1]
                views.append(
                    Structured3DView(self.scene_id, room_id, view_id, p)
                )
        self._views = views
        return self._views

    def get_view_pairs(
        self, room_id: str | None = None
    ) -> list[tuple[Structured3DView, Structured3DView]]:
        """Return pairs of viewpoints in the same room."""
        views = self.get_views()
        by_room: dict[str, list[Structured3DView]] = {}
        for v in views:
            if room_id is not None and v.room_id != room_id:
                continue
            by_room.setdefault(v.room_id, []).append(v)

        pairs: list[tuple[Structured3DView, Structured3DView]] = []
        for r_views in by_room.values():
            n = len(r_views)
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((r_views[i], r_views[j]))
        return pairs

    def __repr__(self) -> str:
        return f"Structured3DScene(id={self.scene_id}, dir={self.scene_dir})"


class Structured3DDataset:
    """
    Main Structured3D Dataset container.

    Auto-locates unpacked Structured3D folder or accepts custom root directory.
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            # Auto-detect default paths
            candidates = [
                ROOT / "Structured3D_perspective_full_00" / "Structured3D",
                ROOT / "Structured3D_perspective_full_00",
                ROOT / "data" / "Structured3D",
            ]
            found: Path | None = None
            for cand in candidates:
                if cand.is_dir() and any(cand.glob("scene_*")):
                    found = cand
                    break
            if found is None:
                raise FileNotFoundError(
                    "Could not auto-locate Structured3D dataset directory. "
                    "Please specify root_dir."
                )
            self.root_dir = found
        else:
            self.root_dir = Path(root_dir)
            if (self.root_dir / "Structured3D").is_dir():
                self.root_dir = self.root_dir / "Structured3D"

        self.scenes: dict[str, Structured3DScene] = {}
        self.views: list[Structured3DView] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Index all scenes and viewpoints."""
        scene_dirs = sorted(self.root_dir.glob("scene_*"))
        for sdir in scene_dirs:
            if sdir.is_dir():
                scene = Structured3DScene(sdir.name, sdir)
                self.scenes[sdir.name] = scene
                self.views.extend(scene.get_views())

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> Structured3DView:
        return self.views[idx]

    def __iter__(self) -> Iterator[Structured3DView]:
        return iter(self.views)

    def get_room_pairs(self) -> list[tuple[Structured3DView, Structured3DView]]:
        """Return all matching viewpoint pairs across the entire dataset."""
        all_pairs: list[tuple[Structured3DView, Structured3DView]] = []
        for scene in self.scenes.values():
            all_pairs.extend(scene.get_view_pairs())
        return all_pairs

    def summary(self) -> dict[str, Any]:
        """Dataset statistics summary."""
        total_scenes = len(self.scenes)
        total_views = len(self.views)
        pairs = self.get_room_pairs()
        return {
            "root_dir": str(self.root_dir),
            "total_scenes": total_scenes,
            "total_viewpoints": total_views,
            "total_matching_pairs": len(pairs),
        }


def visualize_view(
    view: Structured3DView, scene: Structured3DScene | None = None
) -> np.ndarray:
    """
    Create a composite 5-panel visualization:
    [RGB] | [Depth] | [Normal] | [Semantic] | [3D Wireframe Overlay].
    """
    rgb = view.load_rgb(rawlight=True)
    depth_vis = view.colorized_depth()
    normal_vis = view.load_normal()
    semantic_vis = view.colorized_semantic()

    # Create wireframe overlay on RGB
    overlay = rgb.copy()
    cam = view.camera

    # Draw 3D lines if scene is supplied
    if scene is not None:
        lines_3d = scene.get_3d_lines()
        for p1_w, p2_w in lines_3d:
            clipped = cam.project_line_segment(p1_w, p2_w)
            if clipped is not None:
                pt1, pt2 = clipped
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw 2D layout junctions if present
    layout = view.load_layout()
    for junction in layout.get("junctions", []):
        coord = junction.get("coordinate")
        if coord is not None:
            cx, cy = int(round(coord[0])), int(round(coord[1]))
            if 0 <= cx < cam.width and 0 <= cy < cam.height:
                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

    # Label panels
    def put_title(img: np.ndarray, text: str) -> np.ndarray:
        res = img.copy()
        cv2.putText(
            res,
            text,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            res,
            text,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return res

    p1 = put_title(rgb, f"RGB: {view.scene_id} room_{view.room_id} v{view.view_id}")
    p2 = put_title(depth_vis, "Depth Map (Turbo)")
    p3 = put_title(normal_vis, "Surface Normal")
    p4 = put_title(semantic_vis, "Semantic Segmentation (40 classes)")
    p5 = put_title(overlay, "Ground-Truth 3D Wireframe & Layout")

    # Arrange in a 2x3 or 2x2 collage
    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    last_row = np.hstack([p5, np.zeros_like(p5)])

    # Scale down for convenient display
    collage = np.vstack([top_row, bottom_row, last_row])
    scale_factor = 0.5
    h, w = collage.shape[:2]
    collage_small = cv2.resize(
        collage,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_AREA,
    )
    return collage_small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured3D Dataset Reader & Inspector"
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        type=Path,
        default=None,
        help="Structured3D root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--scene", type=str, default=None, help="Scene ID (e.g. scene_00000)"
    )
    parser.add_argument("--room", type=str, default=None, help="Room ID")
    parser.add_argument("--view", type=str, default=None, help="View ID")
    parser.add_argument(
        "--pair",
        action="store_true",
        help="Inspect and visualize a two-view matching pair",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only display dataset summary statistics",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "outputs" / "dataset_preview" / "sample_view.png",
        help="Output image path for visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading Structured3D dataset...")
    dataset = Structured3DDataset(args.dataset_dir)
    stats = dataset.summary()

    print("=" * 60)
    print(f"Dataset root:         {stats['root_dir']}")
    print(f"Total scenes:         {stats['total_scenes']}")
    print(f"Total viewpoints:     {stats['total_viewpoints']}")
    print(f"Total matching pairs: {stats['total_matching_pairs']}")
    print("=" * 60)

    if args.stats_only:
        return

    # Choose a viewpoint to inspect
    selected_scene: Structured3DScene | None = None
    selected_view: Structured3DView | None = None

    if args.scene:
        if args.scene not in dataset.scenes:
            raise KeyError(f"Scene {args.scene} not found in dataset")
        selected_scene = dataset.scenes[args.scene]
        views = selected_scene.get_views()
        if args.room:
            views = [v for v in views if v.room_id == args.room]
        if args.view:
            views = [v for v in views if v.view_id == args.view]
        if not views:
            raise ValueError("No matching views found for criteria")
        selected_view = views[0]
    else:
        # Pick scene_00005 as default if present, else first view
        if "scene_00005" in dataset.scenes:
            selected_scene = dataset.scenes["scene_00005"]
            selected_view = selected_scene.get_views()[0]
        else:
            selected_view = dataset[0]
            selected_scene = dataset.scenes[selected_view.scene_id]

    print(f"\nInspecting Viewpoint:")
    print(f"  Scene:    {selected_view.scene_id}")
    print(f"  Room:     {selected_view.room_id}")
    print(f"  View:     {selected_view.view_id}")
    print(f"  Path:     {selected_view.view_dir}")

    cam = selected_view.camera
    print(f"\nCamera Parameters:")
    print(f"  Position (XYZ mm): {np.round(cam.position, 2)}")
    print(f"  Look Direction:    {np.round(cam.look, 4)}")
    print(f"  Up Direction:      {np.round(cam.up, 4)}")
    print(f"  Focal Length:      fx={cam.fx:.2f}, fy={cam.fy:.2f}")
    print(f"  Principal Point:   cx={cam.cx:.2f}, cy={cam.cy:.2f}")
    print(
        f"  FOV (deg):         h_fov={np.degrees(cam.xfov)*2:.1f}°, v_fov={np.degrees(cam.yfov)*2:.1f}°"
    )

    if args.pair:
        pairs = selected_scene.get_view_pairs(selected_view.room_id)
        if not pairs:
            pairs = dataset.get_room_pairs()
        v1, v2 = pairs[0]
        R_rel, t_rel, rot_deg, base_m = v1.camera.relative_pose_to(v2.camera)
        print(f"\nRelative Pose between View {v1.view_id} and View {v2.view_id}:")
        print(f"  Baseline distance: {base_m:.3f} meters")
        print(f"  Rotation angle:    {rot_deg:.2f} degrees")

        # Side-by-side pair visualization
        img1 = v1.load_rgb()
        img2 = v2.load_rgb()
        cv2.putText(
            img1,
            f"View {v1.view_id}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            img2,
            f"View {v2.view_id} (Rot: {rot_deg:.1f} deg, Base: {base_m:.2f}m)",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
        )
        pair_vis = np.hstack([img1, img2])
        pair_vis_small = cv2.resize(
            pair_vis, (1280, int(720 / 2)), interpolation=cv2.INTER_AREA
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), pair_vis_small)
        print(f"\nSaved pair visualization to: {args.output}")
    else:
        # Generate composite collage
        collage = visualize_view(selected_view, selected_scene)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), collage)
        print(f"\nSaved viewpoint visualization to: {args.output}")


if __name__ == "__main__":
    main()
