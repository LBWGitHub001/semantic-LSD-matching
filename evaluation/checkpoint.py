"""
Checkpoint Resolver & Config Reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple


def resolve_checkpoint_path(ckpt_path: Path | str) -> Tuple[Path, Optional[dict]]:
    """
    Resolves checkpoint path to a valid .pth file and loads any adjacent config.json.
    Supports:
      - Direct path to .pth file
      - Directory containing best_model.pth or last_model.pth
      - 'latest' symlink or directory of timestamped runs
    """
    path = Path(ckpt_path).resolve()
    config_data: Optional[dict] = None

    if path.is_symlink():
        path = path.resolve()

    if path.is_dir():
        # Check for config.json in directory
        if (path / "config.json").is_file():
            try:
                with open(path / "config.json", "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        # If directory has a 'latest' symlink/subdir, check if it contains valid .pth files
        if (path / "latest").exists():
            resolved_latest = (path / "latest").resolve()
            if any(resolved_latest.glob("*.pth")):
                return resolve_checkpoint_path(path / "latest")

        if (path / "best_model.pth").is_file():
            path = path / "best_model.pth"
        elif (path / "last_model.pth").is_file():
            path = path / "last_model.pth"
        else:
            pths = sorted(list(path.glob("**/*.pth")), key=lambda p: p.stat().st_mtime)
            if pths:
                path = pths[-1]
            else:
                raise FileNotFoundError(f"No .pth checkpoint files found in directory: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # If config wasn't found yet, check parent directory
    if config_data is None and (path.parent / "config.json").is_file():
        try:
            with open(path.parent / "config.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            pass

    return path, config_data
