"""Build pair folders of RGB, depth, and DeepLSD distance/angle fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from datasets.homo_matcher_dataset import TARGET_HEIGHT, TARGET_WIDTH
from models.homo_matcher.fields import DEFAULT_WEIGHT, infer_fields, load_deeplsd
from models.homo_matcher.ridge import resize_fields


def _view_dirs(scene_dir: Path) -> list[tuple[str, Path]]:
    found = []
    for rgb in sorted(scene_dir.glob("2D_rendering/*/perspective/full/*/rgb_rawlight.png")):
        view_dir = rgb.parent
        if (view_dir / "depth.png").is_file():
            room = view_dir.parts[-4]
            view = view_dir.parts[-1]
            found.append((f"{room}_v{view}", view_dir))
    return found


def _read_depth(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim == 3:
        raw = raw[..., 0]
    return raw.astype(np.float32) / 1000.0


def build_homo_matcher_dataset(
    out_dir: Path | str,
    s3d_dir: Path | str,
    weight: Path | str = DEFAULT_WEIGHT,
    max_scenes: int | None = None,
    max_pairs: int | None = None,
    image_height: int = TARGET_HEIGHT,
    image_width: int = TARGET_WIDTH,
    device: str | None = None,
) -> dict[str, Any]:
    """Write ``pairs/<id>`` plus train/val/test split files."""
    import torch

    source = Path(s3d_dir)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pairs").mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_deeplsd(Path(weight), torch_device)
    scenes = sorted(path for path in source.glob("scene_*") if path.is_dir())
    if max_scenes is not None:
        scenes = scenes[: max_scenes]
    pair_ids: list[str] = []
    cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def load_view(view_dir: Path):
        if view_dir in cache:
            return cache[view_dir]
        bgr = cv2.imread(str(view_dir / "rgb_rawlight.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(view_dir / "rgb_rawlight.png")
        distance, angle = infer_fields(net, bgr, torch_device)
        depth = _read_depth(view_dir / "depth.png")
        packed = resize_fields(bgr, distance, angle, depth, image_height, image_width)
        cache[view_dir] = packed
        return packed

    for scene in scenes:
        grouped: dict[str, list[tuple[str, Path]]] = {}
        for name, view_dir in _view_dirs(scene):
            room = name.rsplit("_v", 1)[0]
            grouped.setdefault(room, []).append((name, view_dir))
        for room, views in grouped.items():
            views.sort(key=lambda item: int(item[0].rsplit("_v", 1)[1]))
            for (name_a, dir_a), (name_b, dir_b) in zip(views, views[1:]):
                if max_pairs is not None and len(pair_ids) >= max_pairs:
                    break
                pair_id = f"{scene.name}_{name_a}_{name_b}"
                bgr0, df0, angle0, depth0 = load_view(dir_a)
                bgr1, df1, angle1, depth1 = load_view(dir_b)
                pair_dir = destination / "pairs" / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(pair_dir / "image0.png"), bgr0)
                cv2.imwrite(str(pair_dir / "image1.png"), bgr1)
                np.save(pair_dir / "depth0.npy", depth0)
                np.save(pair_dir / "depth1.npy", depth1)
                np.savez_compressed(pair_dir / "fields0.npz", df=df0, angle=angle0)
                np.savez_compressed(pair_dir / "fields1.npz", df=df1, angle=angle1)
                meta = {
                    "pair_id": pair_id,
                    "scene": scene.name,
                    "view0": name_a,
                    "view1": name_b,
                    "height": image_height,
                    "width": image_width,
                }
                (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                pair_ids.append(pair_id)
                print(f"[+] {pair_id}")
            if max_pairs is not None and len(pair_ids) >= max_pairs:
                break
        if max_pairs is not None and len(pair_ids) >= max_pairs:
            break

    count = len(pair_ids)
    if count <= 1:
        splits = {"train": pair_ids, "val": pair_ids, "test": pair_ids}
    elif count == 2:
        splits = {"train": pair_ids[:1], "val": pair_ids[1:], "test": pair_ids[1:]}
    else:
        n_train = max(1, int(round(count * 0.8)))
        n_val = max(1, int(round(count * 0.1)))
        if n_train + n_val >= count:
            n_val = max(1, count - n_train - 1)
            n_train = count - n_val - 1
        splits = {
            "train": pair_ids[:n_train],
            "val": pair_ids[n_train : n_train + n_val],
            "test": pair_ids[n_train + n_val :],
        }
    for name, ids in splits.items():
        payload = {"pairs": ids, "count": len(ids)}
        (destination / f"{name}_split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {"pairs": count, "out_dir": str(destination), **{key: len(value) for key, value in splits.items()}}
    print(f"[+] dataset {summary}")
    return summary
