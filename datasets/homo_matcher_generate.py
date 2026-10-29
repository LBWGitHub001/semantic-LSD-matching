"""Build pair folders of RGB, depth, and DeepLSD distance/angle fields."""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from datasets.homo_matcher_dataset import TARGET_HEIGHT, TARGET_WIDTH
from models.FCN_field.fields import DEFAULT_WEIGHT, infer_fields_batch, load_deeplsd
from models.FCN_field.ridge import resize_fields

_PAIR_FILES = (
    "image0.png",
    "image1.png",
    "depth0.npy",
    "depth1.npy",
    "fields0.npz",
    "fields1.npz",
    "meta.json",
)
_INFER_BATCH = 4


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


def _pair_complete(pair_dir: Path) -> bool:
    return all((pair_dir / name).is_file() for name in _PAIR_FILES)


def _read_view(view_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(view_dir / "rgb_rawlight.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(view_dir / "rgb_rawlight.png")
    return bgr, _read_depth(view_dir / "depth.png")


def _write_pair(destination: Path, record: dict[str, Any], left, right, image_height: int, image_width: int) -> str:
    pair_id = record["pair_id"]
    pair_dir = destination / "pairs" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    bgr0, df0, angle0, depth0 = left
    bgr1, df1, angle1, depth1 = right
    cv2.imwrite(str(pair_dir / "image0.png"), bgr0)
    cv2.imwrite(str(pair_dir / "image1.png"), bgr1)
    np.save(pair_dir / "depth0.npy", depth0)
    np.save(pair_dir / "depth1.npy", depth1)
    np.savez_compressed(pair_dir / "fields0.npz", df=df0, angle=angle0)
    np.savez_compressed(pair_dir / "fields1.npz", df=df1, angle=angle1)
    meta = {
        "pair_id": pair_id,
        "scene": record["scene"],
        "view0": record["view0"],
        "view1": record["view1"],
        "height": image_height,
        "width": image_width,
    }
    (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[+] {pair_id}", flush=True)
    return pair_id


def _enumerate_pairs(scenes: list[Path], max_pairs: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in scenes:
        grouped: dict[str, list[tuple[str, Path]]] = {}
        for name, view_dir in _view_dirs(scene):
            room = name.rsplit("_v", 1)[0]
            grouped.setdefault(room, []).append((name, view_dir))
        for views in grouped.values():
            views.sort(key=lambda item: int(item[0].rsplit("_v", 1)[1]))
            for (name_a, dir_a), (name_b, dir_b) in zip(views, views[1:]):
                if max_pairs is not None and len(records) >= max_pairs:
                    return records
                records.append(
                    {
                        "pair_id": f"{scene.name}_{name_a}_{name_b}",
                        "scene": scene.name,
                        "view0": name_a,
                        "view1": name_b,
                        "dir0": dir_a,
                        "dir1": dir_b,
                    }
                )
    return records


def build_homo_matcher_dataset(
    out_dir: Path | str,
    s3d_dir: Path | str,
    weight: Path | str = DEFAULT_WEIGHT,
    max_scenes: int | None = None,
    max_pairs: int | None = None,
    image_height: int = TARGET_HEIGHT,
    image_width: int = TARGET_WIDTH,
    device: str | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    """Write ``pairs/<id>`` plus train/val/test split files.

    ``num_workers`` threads read images and write pairs. The network stays in
    this process and runs in batches, because a second copy of DeepLSD per
    worker would reload PyTorch onto the same GPU.
    """
    import torch

    source = Path(s3d_dir)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pairs").mkdir(parents=True, exist_ok=True)
    workers = max(1, int(num_workers))
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scenes = sorted(path for path in source.glob("scene_*") if path.is_dir())
    if max_scenes is not None:
        scenes = scenes[: max_scenes]
    records = _enumerate_pairs(scenes, max_pairs)
    pending = [record for record in records if not _pair_complete(destination / "pairs" / record["pair_id"])]
    print(
        f"[*] homo_matcher pairs {len(records)}, already done {len(records) - len(pending)}, "
        f"read/write threads {workers}, GPU batch {_INFER_BATCH} on {torch_device}",
        flush=True,
    )
    net = load_deeplsd(Path(weight), torch_device) if pending else None
    needed: list[Path] = []
    seen: set[Path] = set()
    for record in pending:
        for view_dir in (record["dir0"], record["dir1"]):
            if view_dir not in seen:
                seen.add(view_dir)
                needed.append(view_dir)

    packed: dict[Path, tuple] = {}
    inflight: dict = {}
    cursor = 0
    ready: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def pump_reads(pool: ThreadPoolExecutor) -> None:
        nonlocal cursor
        while cursor < len(needed) and len(inflight) < workers:
            view_dir = needed[cursor]
            inflight[pool.submit(_read_view, view_dir)] = view_dir
            cursor += 1

    def infer_ready(force: bool) -> None:
        if not ready:
            return
        if not force and len(ready) < _INFER_BATCH:
            return
        chunk = list(ready.items())[:_INFER_BATCH]
        for view_dir, _ in chunk:
            del ready[view_dir]
        fields = infer_fields_batch(net, [item[1][0] for item in chunk], torch_device)
        for (view_dir, (bgr, depth)), (distance, angle) in zip(chunk, fields):
            packed[view_dir] = resize_fields(bgr, distance, angle, depth, image_height, image_width)

    write_futures = []
    with ThreadPoolExecutor(max_workers=workers) as read_pool, ThreadPoolExecutor(max_workers=workers) as write_pool:
        pump_reads(read_pool)
        pair_cursor = 0
        while inflight or ready or pair_cursor < len(pending):
            if inflight:
                done, _ = wait(set(inflight), return_when=FIRST_COMPLETED)
                for future in done:
                    view_dir = inflight.pop(future)
                    ready[view_dir] = future.result()
                pump_reads(read_pool)
            infer_ready(force=not inflight)
            while pair_cursor < len(pending):
                record = pending[pair_cursor]
                if record["dir0"] not in packed or record["dir1"] not in packed:
                    break
                write_futures.append(
                    write_pool.submit(
                        _write_pair,
                        destination,
                        record,
                        packed[record["dir0"]],
                        packed[record["dir1"]],
                        image_height,
                        image_width,
                    )
                )
                pair_cursor += 1
                for view_dir in (record["dir0"], record["dir1"]):
                    still_needed = any(
                        item["dir0"] == view_dir or item["dir1"] == view_dir
                        for item in pending[pair_cursor:]
                    )
                    if not still_needed:
                        packed.pop(view_dir, None)
            if not inflight and not ready and pair_cursor < len(pending):
                break
        for future in write_futures:
            future.result()

    pair_ids = [record["pair_id"] for record in records]

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
