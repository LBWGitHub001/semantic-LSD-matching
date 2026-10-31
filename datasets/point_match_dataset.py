"""Planar letter pairs packed as RGB plus DeepLSD distance and angle."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DF_MAX = 5.0


class PointMatchDataset(Dataset):
    """One homography pair. Images are 5-channel tensors, points are aligned by id."""

    def __init__(self, dataset_dir: Path | str, split: str = "train") -> None:
        self.root = Path(dataset_dir)
        split_path = self.root / f"{split}_split.json"
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        self.pair_ids = list(payload["pairs"] if isinstance(payload, dict) else payload)

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, index: int) -> dict:
        pair_id = self.pair_ids[index]
        scene_name, view0, view1 = _parse_pair_id(pair_id)
        scene_dir = self.root / scene_name
        record = _pair_record(scene_dir, view0, view1)
        keypoints = json.loads((scene_dir / "keypoints.json").read_text(encoding="utf-8"))
        by_view = {int(view["view"]): view["points"] for view in keypoints["views"]}
        points0 = []
        points1 = []
        for match in record["matches"]:
            left = by_view[view0][int(match["index0"])]
            right = by_view[view1][int(match["index1"])]
            points0.append([left["x"], left["y"]])
            points1.append([right["x"], right["y"]])
        image0 = _load_view(scene_dir, view0)
        image1 = _load_view(scene_dir, view1)
        return {
            "image0": torch.from_numpy(_pack_input(*image0)),
            "image1": torch.from_numpy(_pack_input(*image1)),
            "homography": torch.tensor(record["H_0_to_1"], dtype=torch.float32),
            "points0": torch.tensor(points0, dtype=torch.float32),
            "points1": torch.tensor(points1, dtype=torch.float32),
            "bgr0": torch.from_numpy(_as_bgr(image0[0])),
            "bgr1": torch.from_numpy(_as_bgr(image1[0])),
            "pair_id": pair_id,
        }


def collate_point_matches(batch: list[dict]) -> dict:
    """Batch size is one pair. Point counts differ, so they are not stacked."""
    item = batch[0]
    return {
        "image0": item["image0"].unsqueeze(0),
        "image1": item["image1"].unsqueeze(0),
        "homography": item["homography"].unsqueeze(0),
        "points0": item["points0"],
        "points1": item["points1"],
        "bgr0": item["bgr0"],
        "bgr1": item["bgr1"],
        "pair_id": item["pair_id"],
    }


def _parse_pair_id(pair_id: str) -> tuple[str, int, int]:
    scene_name, view0, view1 = pair_id.rsplit("_v", 2)
    return scene_name, int(view0), int(view1)


def _pair_record(scene_dir: Path, view0: int, view1: int) -> dict:
    payload = json.loads((scene_dir / "pairs.json").read_text(encoding="utf-8"))
    for record in payload["pairs"]:
        if int(record["view0"]) == view0 and int(record["view1"]) == view1:
            return record
    raise KeyError(f"{scene_dir.name} 没有视角 {view0} 到 {view1} 的配对")


def _load_view(scene_dir: Path, view: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"view_{view:02d}"
    image_path = scene_dir / f"{stem}.png"
    field_path = scene_dir / f"{stem}.npz"
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(image_path)
    if not field_path.is_file():
        raise FileNotFoundError(f"缺少与 {image_path.name} 同名的直线场 {field_path.name}")
    fields = np.load(field_path)
    return image, np.asarray(fields["df"], dtype=np.float32), np.asarray(fields["angle"], dtype=np.float32)


def _pack_input(image: np.ndarray, distance: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """RGB/255, distance in [0, 1] (-log(x+1) normalized), angle/π. A gray letter is copied into three color channels."""
    from models.FCN_field.fields import transform_distance_field

    if image.ndim == 2:
        color = np.stack([image, image, image], axis=0)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color = np.transpose(rgb, (2, 0, 1))
    color = color.astype(np.float32) / 255.0

    dist_arr = np.nan_to_num(distance, nan=0.0).astype(np.float32)
    # 自动兼容新老数据格式:
    # 存量旧数据为原始物理距离 [0, 5.0]，动态应用 -log(x+1) 归一化；
    # 新数据已经归一化为 [0, 1.0]，直接安全截断使用。
    if dist_arr.size > 0 and dist_arr.max() > 1.0 + 1e-4:
        dist_norm = transform_distance_field(dist_arr, max_dist=DF_MAX)
    else:
        dist_norm = np.clip(dist_arr, 0.0, 1.0)

    direction = np.mod(np.nan_to_num(angle, nan=0.0), np.pi) / np.pi
    return np.concatenate([color, dist_norm[None], direction[None]], axis=0).astype(np.float32)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
