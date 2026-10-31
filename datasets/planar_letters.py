"""Planar warps of the letters A–Z for descriptor training.

Each scene draws one letter on blurred noise, then warps that image.
The homography is rotation, translation and scale, plus a small perspective
term. Point ids stay fixed, so the same id in two views is a match.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LETTERS = tuple(chr(ord("A") + index) for index in range(26))
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
)


def build_planar_letters_dataset(
    out_dir: Path | str,
    num_scenes: int = 26,
    views_per_scene: int = 4,
    seed: int = 42,
    preview_scenes: int = 6,
    infer_batch=None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write warped letters, homographies, line fields, pair splits, and a preview."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pair_ids: list[str] = []
    preview_tiles: list[np.ndarray] = []
    if infer_batch is None:
        infer_batch = _load_deeplsd_infer()

    for scene_idx in range(num_scenes):
        letter = LETTERS[scene_idx % len(LETTERS)]
        scene = render_scene(letter, views_per_scene, rng)
        scene_name = f"scene_{scene_idx:05d}"
        scene_dir = out / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        _write_scene(scene_dir, scene)
        _save_line_fields(scene_dir, scene, infer_batch, overwrite=overwrite)
        for pair in scene["pairs"]:
            pair_ids.append(f"{scene_name}_v{pair['view0']}_v{pair['view1']}")
        # 每个字母选取一张代表视角，提取其直线场 (Line Field) 加入大图
        if scene["views"]:
            rep_idx = min(1, len(scene["views"]) - 1)
            shown = scene["views"][rep_idx]
            npz_file = scene_dir / f"view_{rep_idx:02d}.npz"
            if npz_file.is_file():
                fields = np.load(npz_file)
                df = np.asarray(fields["df"], dtype=np.float32)
            else:
                df = np.zeros(shown["image"].shape[:2], dtype=np.float32)
            preview_tiles.append(_draw_line_field_tile(df, letter=letter))

    _write_splits(out, pair_ids)
    if preview_tiles:
        sheet = _tile(preview_tiles, columns=6)
        preview_path = out / "preview.jpg"
        cv2.imwrite(str(preview_path), sheet)
        print(f"[✓] Generated single composite line field preview: {preview_path} ({sheet.shape[1]}x{sheet.shape[0]})")
    print(
        f"[*] planar_letters scenes={num_scenes} views={views_per_scene} "
        f"pairs={len(pair_ids)} -> {out}"
    )
    return {"scenes": num_scenes, "pairs": len(pair_ids)}


def render_scene(letter: str, num_views: int, rng: np.random.Generator) -> dict:
    """One letter, one canonical view, and warped copies. Point ids are shared."""
    height, width = IMAGE_HEIGHT, IMAGE_WIDTH
    mask = _draw_letter(letter, height, width, rng)
    source = _sample_points(mask, rng)
    image0 = _paint(mask, rng)
    views = [
        {
            "view": 0,
            "image": image0,
            "keypoints": [{"id": index, "x": float(x), "y": float(y)} for index, (x, y) in enumerate(source)],
            "H_from_canonical": np.eye(3, dtype=np.float64),
        }
    ]
    for view_idx in range(1, num_views):
        homography = _sample_homography(source, width, height, rng)
        border = float(np.median(image0))
        warped = cv2.warpPerspective(
            image0,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        mapped = _apply_homography(homography, source)
        keypoints = []
        for index, (x, y) in enumerate(mapped):
            if 1.0 <= x < width - 1 and 1.0 <= y < height - 1:
                keypoints.append({"id": index, "x": float(x), "y": float(y)})
        views.append(
            {
                "view": view_idx,
                "image": warped,
                "keypoints": keypoints,
                "H_from_canonical": homography,
            }
        )
    pairs = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            matches = _matches(views[i]["keypoints"], views[j]["keypoints"])
            if len(matches) < 4:
                continue
            homography = views[j]["H_from_canonical"] @ np.linalg.inv(views[i]["H_from_canonical"])
            pairs.append(
                {
                    "view0": i,
                    "view1": j,
                    "matches": matches,
                    "H_0_to_1": (homography / homography[2, 2]).tolist(),
                }
            )
    return {"letter": letter, "width": width, "height": height, "views": views, "pairs": pairs}


def _draw_letter(letter: str, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    font = int(rng.choice(_FONTS))
    thickness = int(rng.integers(8, 14))
    scale = 1.0
    # 字高、字宽封顶都比原来的 0.42 大 50%。
    for _ in range(16):
        (text_w, text_h), baseline = cv2.getTextSize(letter, font, scale, thickness)
        if text_h + baseline >= int(height * 0.63) or text_w >= int(width * 0.63):
            break
        scale *= 1.15
    (text_w, text_h), baseline = cv2.getTextSize(letter, font, scale, thickness)
    origin = ((width - text_w) // 2, (height + text_h) // 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(mask, letter, origin, font, scale, 255, thickness, cv2.LINE_AA)
    return mask


def _sample_points(mask: np.ndarray, rng: np.random.Generator, count: int = 36) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pick = min(count, len(xs))
    chosen = rng.choice(len(xs), size=pick, replace=False)
    return np.stack([xs[chosen], ys[chosen]], axis=1).astype(np.float32)


def _paint(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = mask.shape
    noise = rng.uniform(40, 200, size=(height, width)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.8)
    speckle = rng.normal(0.0, 8.0, size=(height, width)).astype(np.float32)
    image = np.clip(blurred + speckle, 0, 255).astype(np.uint8)
    image[mask > 0] = 22
    return image


def _sample_homography(
    source: np.ndarray, width: int, height: int, rng: np.random.Generator
) -> np.ndarray:
    """Rotation, translation and scale about the image center, plus a little perspective.

    Reject a draw when letter points leave the frame or pass through the horizon.
    """
    center = np.array([[1.0, 0.0, -width / 2], [0.0, 1.0, -height / 2], [0.0, 0.0, 1.0]])
    back = np.array([[1.0, 0.0, width / 2], [0.0, 1.0, height / 2], [0.0, 0.0, 1.0]])
    for _ in range(30):
        angle = float(rng.uniform(-0.8, 0.8))
        scale_x = float(rng.uniform(0.75, 1.25))
        scale_y = scale_x * float(rng.uniform(0.9, 1.1))
        shift_x = float(rng.uniform(-28, 28))
        shift_y = float(rng.uniform(-22, 22))
        perspective = np.eye(3)
        perspective[2, 0] = float(rng.uniform(-1.5e-4, 1.5e-4))
        perspective[2, 1] = float(rng.uniform(-1.5e-4, 1.5e-4))
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(
            [[scale_x * cosine, -scale_y * sine, shift_x], [scale_x * sine, scale_y * cosine, shift_y], [0.0, 0.0, 1.0]]
        )
        homography = back @ perspective @ rotation @ center
        homography = homography / homography[2, 2]
        if len(source) == 0:
            return homography
        mapped = _apply_homography(homography, source)
        inside = (
            (mapped[:, 0] > 2)
            & (mapped[:, 0] < width - 3)
            & (mapped[:, 1] > 2)
            & (mapped[:, 1] < height - 3)
        )
        weights = _homogeneous_weights(homography, source)
        if int(inside.sum()) >= max(8, int(0.55 * len(source))) and np.all(weights > 0.35):
            return homography
    return np.eye(3, dtype=np.float64)


def _apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    homo = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    mapped = homo @ homography.T
    return mapped[:, :2] / mapped[:, 2:3]


def _homogeneous_weights(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    return (homo @ homography.T)[:, 2]


def _matches(first: list[dict], second: list[dict]) -> list[dict[str, int]]:
    index_in_second = {int(point["id"]): index for index, point in enumerate(second)}
    matches = []
    for index, point in enumerate(first):
        point_id = int(point["id"])
        if point_id in index_in_second:
            matches.append({"id": point_id, "index0": index, "index1": index_in_second[point_id]})
    return matches


def _write_scene(scene_dir: Path, scene: dict) -> None:
    for view in scene["views"]:
        cv2.imwrite(str(scene_dir / f"view_{view['view']:02d}.png"), view["image"])
    keypoints = {"letter": scene["letter"], "views": [{"view": view["view"], "points": view["keypoints"]} for view in scene["views"]]}
    transforms = {
        "letter": scene["letter"],
        "width": scene["width"],
        "height": scene["height"],
        "views": [],
    }
    for view in scene["views"]:
        forward = view["H_from_canonical"]
        inverse = np.linalg.inv(forward)
        transforms["views"].append(
            {
                "view": view["view"],
                "H_from_canonical": (forward / forward[2, 2]).tolist(),
                "H_to_canonical": (inverse / inverse[2, 2]).tolist(),
            }
        )
    pairs = {"letter": scene["letter"], "pairs": scene["pairs"]}
    (scene_dir / "keypoints.json").write_text(json.dumps(keypoints), encoding="utf-8")
    (scene_dir / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    (scene_dir / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")


def _save_line_fields(scene_dir: Path, scene: dict, infer_batch, overwrite: bool = False) -> None:
    """Write view_XX.npz beside view_XX.png. An existing file is left alone unless overwrite=True."""
    from models.FCN_field.fields import transform_distance_field

    pending_paths: list[Path] = []
    pending_bgr: list[np.ndarray] = []
    for view in scene["views"]:
        dest = scene_dir / f"view_{view['view']:02d}.npz"
        if dest.is_file() and not overwrite:
            continue
        pending_paths.append(dest)
        pending_bgr.append(_as_bgr(view["image"]))
    if not pending_bgr:
        return
    fields = infer_batch(pending_bgr)
    for dest, (distance, angle) in zip(pending_paths, fields):
        df_norm = transform_distance_field(distance, max_dist=5.0)
        np.savez_compressed(
            dest,
            df=df_norm,
            angle=np.asarray(angle, dtype=np.float32),
        )


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _load_deeplsd_infer():
    import torch
    from models.FCN_field.fields import DEFAULT_WEIGHT, infer_fields_batch, load_deeplsd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_deeplsd(DEFAULT_WEIGHT, device)

    def infer_batch(bgrs: list[np.ndarray]):
        return infer_fields_batch(net, bgrs, device)

    return infer_batch


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


def _draw_line_field_tile(df: np.ndarray, letter: str = "") -> np.ndarray:
    """Render the normalized distance field as a high-contrast line field heatmap."""
    df_norm = np.clip(np.nan_to_num(df, nan=0.0), 0.0, 1.0)
    df_u8 = (df_norm * 255.0).astype(np.uint8)
    canvas = cv2.applyColorMap(df_u8, cv2.COLORMAP_MAGMA)
    if letter:
        cv2.rectangle(canvas, (8, 8), (48, 36), (15, 15, 15), -1)
        cv2.putText(canvas, letter, (16, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (60, 60, 60), 1)
    return canvas


def _draw_keypoints(image: np.ndarray, keypoints: list[dict], letter: str = "") -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for point in keypoints:
        center = (int(round(point["x"])), int(round(point["y"])))
        cv2.circle(canvas, center, 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, 2, (0, 230, 0), -1, lineType=cv2.LINE_AA)
    if letter:
        cv2.rectangle(canvas, (8, 8), (48, 36), (15, 15, 15), -1)
        cv2.putText(canvas, letter, (16, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), (60, 60, 60), 1)
    return canvas


def _tile(tiles: list[np.ndarray], columns: int = 6) -> np.ndarray:
    """Composite letter tiles into a single grid sheet (default 6 columns)."""
    if not tiles:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    columns = min(columns, max(len(tiles), 1))
    height, width = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile

    # 若有多余空格（例如 26 个字母放在 6 列 5 行中会有 4 个多余格子），填充精美的数据集信息横幅
    empty_slots = rows * columns - len(tiles)
    if empty_slots > 0:
        r, c_start = divmod(len(tiles), columns)
        bx0 = c_start * width
        bx1 = columns * width
        by0 = r * height
        by1 = (r + 1) * height
        canvas[by0:by1, bx0:bx1] = (25, 25, 25)
        cv2.rectangle(canvas, (bx0, by0), (bx1 - 1, by1 - 1), (60, 60, 60), 1)

        cv2.putText(
            canvas,
            "Planar Letters - Line Field Preview (A - Z)",
            (bx0 + 40, by0 + int(height * 0.35)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{len(tiles)} Letters | DeepLSD Line Distance Field (-log(x+1)) [0.0 ~ 1.0]",
            (bx0 + 40, by0 + int(height * 0.58)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 210, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Ridge / Line Center: 1.0 (Bright Yellow-White) | Background: 0.0 (Dark)",
            (bx0 + 40, by0 + int(height * 0.75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
    return canvas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate planar letters dataset with DeepLSD fields.")
    parser.add_argument("--out", type=str, default="data/planar_letters", help="Output directory")
    parser.add_argument("--scenes", type=int, default=26, help="Number of letter scenes (default: 26)")
    parser.add_argument("--views", type=int, default=4, help="Views per scene (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files")
    args = parser.parse_args()

    print(f"[*] Building planar letters dataset in {args.out} (scenes={args.scenes}, views={args.views}, overwrite={args.overwrite})...")
    build_planar_letters_dataset(
        out_dir=args.out,
        num_scenes=args.scenes,
        views_per_scene=args.views,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("[+] Done!")
