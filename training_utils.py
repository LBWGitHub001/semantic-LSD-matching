"""Shared Training Utilities and Infrastructure for all Matching Models."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import torch

TQDM_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def safe_load_checkpoint(path: Path | str, map_location: Any = "cpu") -> dict[str, Any]:
    """Safely loads a torch checkpoint handling PyTorch 2.6+ weights_only semantics."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_git_commit() -> str:
    """Retrieves short git commit hash if available, otherwise returns 'unknown'."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def save_config_json(config_path: Path, config_data: dict[str, Any]) -> None:
    """Atomically writes configuration and experiment metadata as both JSON and YAML."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        from configs import save_config
        save_config(config_data, config_path.with_suffix(".yaml"))
    except Exception as e:
        print(f"[!] Warning: Failed to write {config_path.name}: {e}")


def update_latest_symlink(base_dir: Path, target_dir: Path) -> None:
    """Updates a 'latest' symlink in base_dir pointing to target_dir."""
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        latest_link = base_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(target_dir.name, target_is_directory=True)
    except Exception:
        pass


def worker_init_fn(worker_id: int) -> None:
    """Disables OpenCV internal multi-threading within worker processes to prevent thread contention."""
    cv2.setNumThreads(0)


def create_experiment_dir(
    base_save_dir: str | Path,
    model_name: str,
    label: str | None = None,
    no_timestamp: bool = False,
    run_name: str | None = None,
) -> tuple[Path, str]:
    """Creates a timestamped or custom experiment directory and updates the latest symlink."""
    base_p = Path(base_save_dir)
    if no_timestamp:
        save_dir = base_p
        r_name = base_p.name
    elif run_name:
        r_name = run_name
        save_dir = base_p / r_name
    else:
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_tag = f"+【{label}】" if label else ""
        r_name = f"{model_name}{label_tag}+【{start_time_str}】"
        save_dir = base_p / r_name

    save_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(base_p, save_dir)
    return save_dir, r_name


def build_object_forward_kwargs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move optional line and bbox inputs in a region/object batch to device."""
    kwargs: dict[str, Any] = {}
    for key0, key1 in (("patch_lines0", "patch_lines1"), ("line_mask0", "line_mask1"), ("bboxes0", "bboxes1")):
        if batch.get(key0) is not None and batch.get(key1) is not None:
            kwargs[key0] = batch[key0].to(device)
            kwargs[key1] = batch[key1].to(device)
    return kwargs


__all__ = [
    "TQDM_BAR_FORMAT",
    "safe_load_checkpoint",
    "get_git_commit",
    "save_config_json",
    "update_latest_symlink",
    "worker_init_fn",
    "create_experiment_dir",
    "build_object_forward_kwargs",
]
