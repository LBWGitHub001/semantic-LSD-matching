"""
Dataset Factory: Auto-Configures Dataset & DataLoader for Evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from datasets import (
    ObjectMatchingDataset,
    collate_single_pair,
    SemanticRegionDataset,
    collate_region_pairs,
)
from models import uses_patch_lines


class DatasetFactory:
    """Instantiates the appropriate dataset and collate function based on model and directory."""

    @staticmethod
    def create(
        model_info: Dict[str, Any],
        data_dir: Path | str,
        split: str = "test",
        min_area_ratio: float = 0.50,
        num_workers: int = 2,
    ) -> Tuple[DataLoader, bool]:
        p_data = Path(data_dir).resolve()
        use_region_ds = False

        if (p_data / "train_split.json").is_file() or (p_data / f"{split}_split.json").is_file():
            use_region_ds = True
        elif any((d / "correspondence.json").is_file() for d in p_data.iterdir() if d.is_dir()):
            use_region_ds = True

        if model_info["family"] == "homography":
            from datasets.homo_matcher_dataset import HomoMatcherDataset, collate_homo_matcher

            dataset = HomoMatcherDataset(dataset_dir=p_data, split=split)
            collate_fn = collate_homo_matcher
        elif model_info["family"] == "object":
            if use_region_ds:
                model_type = model_info.get("model_type", "xfeat_object")
                needs_lines = uses_patch_lines(model_type) if model_type else False
                dataset = SemanticRegionDataset(
                    dataset_dir=p_data,
                    split=split,
                    min_area_ratio=min_area_ratio,
                    load_masks=False,  # Lightweight fast loading
                    load_lines=needs_lines,
                )
                collate_fn = collate_region_pairs
            else:
                dataset = ObjectMatchingDataset(data_dir=p_data, split=split)
                collate_fn = collate_single_pair
        else:
            # Line matching dataset fallback
            from datasets import LineFeatPairDataset
            dataset = LineFeatPairDataset(dataset_dir=p_data, split=split)
            collate_fn = None

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return loader, use_region_ds
