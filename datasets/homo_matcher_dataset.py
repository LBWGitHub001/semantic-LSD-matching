"""Pairs of Structured3D views with DeepLSD fields and depth."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.FCN_field.ridge import extract_ridge_lines, pack_input, resize_fields

TARGET_HEIGHT = 384
TARGET_WIDTH = 640


class HomoMatcherDataset(Dataset):
    """Loads 5-channel views, depth, and ridge samples for one pair."""

    def __init__(
        self,
        dataset_dir: Path | str,
        split: str = "train",
        sample_points: int = 16,
        max_lines: int = 24,
        image_height: int = TARGET_HEIGHT,
        image_width: int = TARGET_WIDTH,
    ) -> None:
        self.root = Path(dataset_dir)
        split_path = self.root / f"{split}_split.json"
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        self.pair_ids = list(payload["pairs"] if isinstance(payload, dict) else payload)
        self.sample_points = int(sample_points)
        self.max_lines = int(max_lines)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, index: int) -> dict:
        pair_id = self.pair_ids[index]
        pair_dir = self.root / "pairs" / pair_id
        bgr0 = _read_color(pair_dir / "image0.png")
        bgr1 = _read_color(pair_dir / "image1.png")
        depth0 = np.load(pair_dir / "depth0.npy")
        depth1 = np.load(pair_dir / "depth1.npy")
        fields0 = np.load(pair_dir / "fields0.npz")
        fields1 = np.load(pair_dir / "fields1.npz")
        bgr0, df0, angle0, depth0 = resize_fields(
            bgr0, fields0["df"], fields0["angle"], depth0, self.image_height, self.image_width
        )
        bgr1, df1, angle1, depth1 = resize_fields(
            bgr1, fields1["df"], fields1["angle"], depth1, self.image_height, self.image_width
        )
        _, samples = extract_ridge_lines(
            df0,
            angle0,
            sample_points=self.sample_points,
            max_lines=self.max_lines,
        )
        ridges = np.zeros((self.max_lines, self.sample_points, 2), dtype=np.float32)
        mask = np.zeros((self.max_lines,), dtype=bool)
        count = min(len(samples), self.max_lines)
        if count:
            ridges[:count] = samples[:count]
            mask[:count] = True
        return {
            "image0": torch.from_numpy(pack_input(bgr0, df0, angle0)),
            "image1": torch.from_numpy(pack_input(bgr1, df1, angle1)),
            "depth0": torch.from_numpy(np.asarray(depth0, dtype=np.float32)).unsqueeze(0),
            "depth1": torch.from_numpy(np.asarray(depth1, dtype=np.float32)).unsqueeze(0),
            "ridges0": torch.from_numpy(ridges),
            "ridge_mask": torch.from_numpy(mask),
            "bgr0": torch.from_numpy(bgr0),
            "bgr1": torch.from_numpy(bgr1),
            "pair_id": pair_id,
        }


def collate_homo_matcher(batch: list[dict]) -> dict:
    stacked = {}
    for key in ("image0", "image1", "depth0", "depth1", "ridges0", "ridge_mask", "bgr0", "bgr1"):
        stacked[key] = torch.stack([item[key] for item in batch], dim=0)
    stacked["pair_id"] = [item["pair_id"] for item in batch]
    return stacked


def _read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image
