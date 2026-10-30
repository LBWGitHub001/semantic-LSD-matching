"""Synthetic geometric shapes for descriptor training.

One shape sits in 3D. Several cameras look at it. The background is blurred
noise. Corner ids are stable, so the same id in two views is a match.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from datasets.generators.procedural_room import create_look_at

CATEGORIES = ("triangle", "quad", "cube", "star", "lines", "checkerboard", "quad_grid")
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320


def build_descriptor_shapes_dataset(
    out_dir: Path | str,
    num_scenes: int = 20,
    views_per_scene: int = 4,
    seed: int = 42,
    preview_scenes: int = len(CATEGORIES),
) -> dict[str, int]:
    """Write scenes, pair splits, and a green-dot preview."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pair_ids: list[str] = []
    preview_tiles: list[np.ndarray] = []

    for scene_idx in range(num_scenes):
        category = CATEGORIES[scene_idx % len(CATEGORIES)]
        scene = render_scene(category, views_per_scene, rng)
        scene_name = f"scene_{scene_idx:05d}"
        scene_dir = out / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        _write_scene(scene_dir, scene)
        for pair in scene["pairs"]:
            pair_ids.append(f"{scene_name}_v{pair['view0']}_v{pair['view1']}")
        if scene_idx < preview_scenes and scene["views"]:
            preview_tiles.append(_draw_keypoints(scene["views"][0]["image"], scene["views"][0]["keypoints"]))

    _write_splits(out, pair_ids)
    if preview_tiles:
        cv2.imwrite(str(out / "preview.jpg"), _tile(preview_tiles))
    print(
        f"[*] descriptor_shapes scenes={num_scenes} views={views_per_scene} "
        f"pairs={len(pair_ids)} -> {out}"
    )
    return {"scenes": num_scenes, "pairs": len(pair_ids)}


def render_scene(category: str, num_views: int, rng: np.random.Generator) -> dict:
    """Render one shape from several viewpoints. Keypoint ids are corner indices."""
    points, faces, edges, shade, corner_count = _make_shape(category, rng)
    height, width = IMAGE_HEIGHT, IMAGE_WIDTH
    intrinsics = _intrinsics(width, height)
    views = []
    for view_idx, eye in enumerate(_camera_eyes(num_views, rng)):
        rotation, translation = create_look_at(eye.astype(np.float32), np.zeros(3, dtype=np.float32))
        image, depth = _render_view(points, faces, edges, shade, intrinsics, rotation, translation, rng)
        keypoints = _visible_keypoints(
            points, intrinsics, rotation, translation, depth, corner_count
        )
        views.append(
            {
                "view": view_idx,
                "image": image,
                "keypoints": keypoints,
                "R": rotation,
                "t": translation,
                "eye": eye.astype(np.float32),
            }
        )
    pairs = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            matches = _matches(views[i]["keypoints"], views[j]["keypoints"])
            if len(matches) < 3:
                continue
            pairs.append({"view0": i, "view1": j, "matches": matches})
    return {
        "category": category,
        "width": width,
        "height": height,
        "K": intrinsics,
        "points_world": points,
        "views": views,
        "pairs": pairs,
    }


def _make_shape(
    category: str, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int]], np.ndarray, int]:
    if category == "triangle":
        points, faces, edges = _polygon(3, rng)
    elif category == "quad":
        points, faces, edges = _polygon(4, rng)
    elif category == "cube":
        points, faces, edges = _cube(rng)
    elif category == "star":
        points, faces, edges = _star(rng)
    elif category == "lines":
        points, faces, edges = _lines(rng)
    elif category == "checkerboard":
        points, faces, edges = _grid(rng, filled=True)
    elif category == "quad_grid":
        points, faces, edges = _grid(rng, filled=False)
    else:
        raise ValueError(f"unknown shape {category}")
    # The last vertex of a star is only there to fill the polygon. It is not a corner.
    corner_count = len(points) - 1 if category == "star" else len(points)
    points = _tilt(points, rng)
    shade = _checker_shade(faces) if category == "checkerboard" else _face_shade(points, faces)
    return points.astype(np.float32), faces, edges, shade, corner_count


def _polygon(count: int, rng: np.random.Generator) -> tuple[np.ndarray, list, list]:
    radius = rng.uniform(0.45, 0.8, size=count).astype(np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + rng.uniform(-0.2, 0.2)
    points = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)], axis=1)
    faces = [(0, i, i + 1) for i in range(1, count - 1)]
    edges = [(i, (i + 1) % count) for i in range(count)]
    return points.astype(np.float32), faces, edges


def _cube(rng: np.random.Generator) -> tuple[np.ndarray, list, list]:
    extent = float(rng.uniform(0.38, 0.58))
    coords = [-extent, extent]
    points = np.array([[x, y, z] for x in coords for y in coords for z in coords], dtype=np.float32)
    # Indices: x changes slowest, then y, then z.
    quads = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    faces = []
    edges = []
    for a, b, c, d in quads:
        faces.append((a, b, c))
        faces.append((a, c, d))
        edges.extend([(a, b), (b, c), (c, d), (d, a)])
    return points, faces, edges


def _star(rng: np.random.Generator) -> tuple[np.ndarray, list, list]:
    """Ten outline corners plus a center used only to fill the concave polygon."""
    outer = float(rng.uniform(0.62, 0.85))
    inner = outer * 0.42
    outline = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -np.pi / 2 + index * np.pi / 5
        outline.append([radius * np.cos(angle), radius * np.sin(angle), 0.0])
    points = np.array(outline + [[0.0, 0.0, 0.0]], dtype=np.float32)
    center = 10
    faces = [(center, index, (index + 1) % 10) for index in range(10)]
    edges = [(index, (index + 1) % 10) for index in range(10)]
    return points, faces, edges


def _lines(rng: np.random.Generator) -> tuple[np.ndarray, list, list]:
    segments = []
    for _ in range(4):
        start = rng.uniform(-0.75, 0.75, size=2)
        stop = rng.uniform(-0.75, 0.75, size=2)
        if np.linalg.norm(stop - start) < 0.35:
            stop = start + np.array([0.55, 0.25])
        segments.append((start.astype(np.float32), stop.astype(np.float32)))
    cuts: list[list[float]] = [[0.0, 1.0] for _ in segments]
    for i, (start, stop) in enumerate(segments):
        for j in range(i + 1, len(segments)):
            other_start, other_stop = segments[j]
            left, right = _segment_params(start, stop, other_start, other_stop)
            if left is None:
                continue
            if 0.04 < left < 0.96 and 0.04 < right < 0.96:
                cuts[i].append(float(left))
                cuts[j].append(float(right))
    points: list[np.ndarray] = []
    edges = []

    def _add(point: np.ndarray) -> int:
        for index, existing in enumerate(points):
            if np.linalg.norm(existing - point) < 1e-3:
                return index
        points.append(point.astype(np.float32))
        return len(points) - 1

    for (start, stop), times in zip(segments, cuts):
        ordered = sorted(set(np.round(times, 5)))
        ids = [_add((1.0 - t) * start + t * stop) for t in ordered]
        edges.extend((ids[k], ids[k + 1]) for k in range(len(ids) - 1) if ids[k] != ids[k + 1])
    packed = np.zeros((len(points), 3), dtype=np.float32)
    packed[:, :2] = np.stack(points, axis=0)
    return packed, [], edges


def _grid(rng: np.random.Generator, filled: bool) -> tuple[np.ndarray, list, list]:
    cells = int(rng.integers(3, 6))
    span = float(rng.uniform(0.55, 0.85))
    axis = np.linspace(-span, span, cells + 1, dtype=np.float32)
    points = np.array([[x, y, 0.0] for y in axis for x in axis], dtype=np.float32)
    stride = cells + 1
    faces = []
    edges = []
    for row in range(cells):
        for col in range(cells):
            a = row * stride + col
            b = a + 1
            c = a + stride + 1
            d = a + stride
            if filled:
                faces.append((a, b, c))
                faces.append((a, c, d))
            edges.extend([(a, b), (b, c), (c, d), (d, a)])
    return points, faces, edges


def _tilt(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3).astype(np.float64)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = float(rng.uniform(-0.5, 0.5))
    rotation, _ = cv2.Rodrigues(axis * angle)
    return (points.astype(np.float64) @ rotation.T).astype(np.float32)


def _face_shade(points: np.ndarray, faces: list[tuple[int, int, int]]) -> np.ndarray:
    """Gray level per face. Checkerboards alternate; other faces follow the normal."""
    light = np.array([0.25, 0.7, 0.4], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = np.zeros(len(faces), dtype=np.uint8)
    for index, (a, b, c) in enumerate(faces):
        normal = np.cross(points[b] - points[a], points[c] - points[a])
        length = np.linalg.norm(normal)
        if length < 1e-6:
            shade[index] = 160
            continue
        normal /= length
        facing = abs(float(np.dot(normal, light)))
        shade[index] = np.uint8(np.clip(70 + 150 * facing, 40, 220))
    return shade


def _checker_shade(faces: list) -> np.ndarray:
    cells = int(round(np.sqrt(len(faces) / 2)))
    shade = np.zeros(len(faces), dtype=np.uint8)
    for index in range(0, len(faces), 2):
        cell = index // 2
        row, col = divmod(cell, cells)
        gray = 35 if (row + col) % 2 == 0 else 210
        shade[index] = gray
        if index + 1 < len(shade):
            shade[index + 1] = gray
    return shade


def _camera_eyes(num_views: int, rng: np.random.Generator, radius: float = 2.7) -> list[np.ndarray]:
    azimuths = np.linspace(-0.95, 0.95, num_views)
    eyes = []
    for azimuth in azimuths:
        azimuth = float(azimuth + rng.uniform(-0.06, 0.06))
        elevation = float(rng.uniform(0.35, 0.95))
        eye = np.array(
            [
                radius * np.cos(elevation) * np.sin(azimuth),
                radius * np.sin(elevation) * 0.75 + 0.2,
                radius * np.cos(elevation) * np.cos(azimuth),
            ],
            dtype=np.float32,
        )
        eyes.append(eye)
    return eyes


def _intrinsics(width: int, height: int) -> np.ndarray:
    focal = 300.0
    return np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _render_view(
    points: np.ndarray,
    faces: list[tuple[int, int, int]],
    edges: list[tuple[int, int]],
    shade: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    image = _noise_background(height, width, rng)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    camera_points = (rotation @ points.T).T + translation
    for face, gray in zip(faces, shade):
        triangle = camera_points[list(face)]
        if np.any(triangle[:, 2] <= 0.12):
            continue
        pixels = _project_camera(triangle, intrinsics)
        _rasterize(image, depth, pixels, int(gray))
    for start, stop in edges:
        _stamp_edge(image, depth, camera_points[start], camera_points[stop], intrinsics)
    return image, depth


def _noise_background(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.uniform(25, 215, size=(height, width)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.8)
    speckle = rng.normal(0.0, 8.0, size=(height, width)).astype(np.float32)
    return np.clip(blurred + speckle, 0, 255).astype(np.uint8)


def _project_camera(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = points[:, 2:3]
    pixels = np.concatenate([(points[:, :2] / z), np.ones_like(z)], axis=1)
    pixels = pixels @ intrinsics.T
    pixels[:, 2] = points[:, 2]
    return pixels


def _rasterize(image: np.ndarray, depth: np.ndarray, pixels: np.ndarray, gray: int) -> None:
    v0, v1, v2 = pixels
    min_x = max(int(np.floor(min(v0[0], v1[0], v2[0]))), 0)
    max_x = min(int(np.ceil(max(v0[0], v1[0], v2[0]))), image.shape[1] - 1)
    min_y = max(int(np.floor(min(v0[1], v1[1], v2[1]))), 0)
    max_y = min(int(np.ceil(max(v0[1], v1[1], v2[1]))), image.shape[0] - 1)
    if min_x > max_x or min_y > max_y:
        return
    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(float(denom)) < 1e-6:
        return
    w0 = ((v1[1] - v2[1]) * (grid_x - v2[0]) + (v2[0] - v1[0]) * (grid_y - v2[1])) / denom
    w1 = ((v2[1] - v0[1]) * (grid_x - v2[0]) + (v0[0] - v2[0]) * (grid_y - v2[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    zz = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
    view = depth[min_y : max_y + 1, min_x : max_x + 1]
    closer = inside & (zz < view)
    view[closer] = zz[closer]
    image[min_y : max_y + 1, min_x : max_x + 1][closer] = gray


def _stamp_edge(
    image: np.ndarray,
    depth: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    if start[2] <= 0.12 or stop[2] <= 0.12:
        return
    samples = np.linspace(0.0, 1.0, 80, dtype=np.float32)[:, None]
    camera = start[None, :] * (1.0 - samples) + stop[None, :] * samples
    pixels = _project_camera(camera, intrinsics)
    height, width = depth.shape
    for x, y, z in pixels:
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                px = ix + ox
                py = iy + oy
                if px < 0 or py < 0 or px >= width or py >= height:
                    continue
                if z <= depth[py, px] + 0.02:
                    depth[py, px] = min(float(depth[py, px]), float(z))
                    image[py, px] = 18


def _visible_keypoints(
    points: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    depth: np.ndarray,
    corner_count: int,
) -> list[dict[str, float]]:
    """A corner is kept when its pixel lands on geometry at the same depth."""
    camera = (rotation @ points.T).T + translation
    pixels = _project_camera(camera, intrinsics)
    height, width = depth.shape
    kept = []
    for point_id, (x, y, z) in enumerate(pixels[:corner_count]):
        if z <= 0.12:
            continue
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        if ix < 2 or iy < 2 or ix >= width - 2 or iy >= height - 2:
            continue
        patch = depth[iy - 2 : iy + 3, ix - 2 : ix + 3]
        finite = patch[np.isfinite(patch)]
        if finite.size == 0:
            continue
        # 附近有与该角点同样深度的像素，说明它露在表面上。被挡住时，缓冲里只剩更近的面。
        if np.any(np.abs(finite - z) <= max(0.05, 0.04 * float(z))):
            kept.append({"id": point_id, "x": float(x), "y": float(y)})
    return kept


def _matches(first: list[dict], second: list[dict]) -> list[dict[str, int]]:
    index_in_second = {int(point["id"]): index for index, point in enumerate(second)}
    matches = []
    for index, point in enumerate(first):
        point_id = int(point["id"])
        if point_id in index_in_second:
            matches.append({"id": point_id, "index0": index, "index1": index_in_second[point_id]})
    return matches


def _segment_params(
    start: np.ndarray, stop: np.ndarray, other_start: np.ndarray, other_stop: np.ndarray
) -> tuple[float | None, float | None]:
    direction = stop - start
    other = other_stop - other_start
    denom = float(direction[0] * other[1] - direction[1] * other[0])
    if abs(denom) < 1e-8:
        return None, None
    offset = other_start - start
    left = float(offset[0] * other[1] - offset[1] * other[0]) / denom
    right = float(offset[0] * direction[1] - offset[1] * direction[0]) / denom
    return left, right


def _write_scene(scene_dir: Path, scene: dict) -> None:
    for view in scene["views"]:
        cv2.imwrite(str(scene_dir / f"view_{view['view']:02d}.png"), view["image"])
    keypoints = {
        "views": [
            {"view": view["view"], "points": view["keypoints"]}
            for view in scene["views"]
        ]
    }
    cameras = {
        "K": scene["K"].tolist(),
        "width": scene["width"],
        "height": scene["height"],
        "category": scene["category"],
        "points_world": scene["points_world"].tolist(),
        "views": {
            f"view_{view['view']:02d}": {
                "R": view["R"].tolist(),
                "t": view["t"].tolist(),
                "eye": view["eye"].tolist(),
            }
            for view in scene["views"]
        },
    }
    pairs = {"category": scene["category"], "pairs": scene["pairs"]}
    (scene_dir / "keypoints.json").write_text(json.dumps(keypoints), encoding="utf-8")
    (scene_dir / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
    (scene_dir / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")


def _write_splits(out: Path, pair_ids: list[str]) -> None:
    count = len(pair_ids)
    if count == 0:
        train, val, test = [], [], []
    elif count < 3:
        train, val, test = list(pair_ids), list(pair_ids), list(pair_ids)
    else:
        n_train = max(int(count * 0.8), 1)
        n_val = max(int(count * 0.1), 1)
        train = pair_ids[:n_train]
        val = pair_ids[n_train : n_train + n_val]
        test = pair_ids[n_train + n_val :] or pair_ids[-1:]
    for name, rows in (("train_split.json", train), ("val_split.json", val), ("test_split.json", test)):
        (out / name).write_text(json.dumps({"pairs": rows, "count": len(rows)}, indent=2), encoding="utf-8")


def _draw_keypoints(image: np.ndarray, keypoints: list[dict]) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in keypoints:
        center = (int(round(point["x"])), int(round(point["y"])))
        cv2.circle(canvas, center, 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return canvas


def _tile(tiles: list[np.ndarray]) -> np.ndarray:
    columns = 3
    height, width = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile
    return canvas
