"""
Unified Configuration Management System.

Provides YAML-based configuration loading, saving, validation,
and seamless integration with argparse CLI entrypoints across:
  - Dataset generation (generate_dataset.py)
  - Model training (train.py)
  - Evaluation & Benchmarking (eval.py)
  - Unified pipeline execution (run.py)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent

# Canonical model name to YAML file mapping
MODEL_CONFIG_MAP: Dict[str, str] = {
    "superpoint": "superpoint.yaml",
    "xfeat_object": "xfeat_object.yaml",
    "xfeat_line_matcher": "xfeat_line_matcher.yaml",
    "mask_line": "mask_line.yaml",
}

# Aliases
MODEL_ALIAS_MAP: Dict[str, str] = {
    "xfeat": "xfeat_object",
    "object": "xfeat_object",
    "xfeat_line": "xfeat_line_matcher",
    "line_object": "xfeat_line_matcher",
    "mask": "mask_line",
}


def get_model_config_path(model_name: str) -> Path:
    """Resolve a model name or alias to its corresponding YAML config path."""
    clean = model_name.strip().lower()
    canonical = MODEL_ALIAS_MAP.get(clean, clean)
    if canonical in MODEL_CONFIG_MAP:
        return CONFIGS_DIR / MODEL_CONFIG_MAP[canonical]
    raise ValueError(
        f"Unknown model config '{model_name}'. Available: {list(MODEL_CONFIG_MAP.keys())}"
    )


def load_config(path_or_dict: Path | str | dict) -> dict[str, Any]:
    """Load and parse a YAML configuration file or validate an existing dictionary.

    Returns the parsed dictionary with path validations.
    """
    if isinstance(path_or_dict, dict):
        return path_or_dict

    config_path = Path(path_or_dict)
    if not config_path.is_absolute():
        # Check relative to REPO_ROOT first, then CONFIGS_DIR
        if (REPO_ROOT / config_path).is_file():
            config_path = (REPO_ROOT / config_path).resolve()
        elif (CONFIGS_DIR / config_path).is_file():
            config_path = (CONFIGS_DIR / config_path).resolve()
        else:
            config_path = config_path.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config structure in {config_path}: expected dictionary root")

    data["_config_path"] = str(config_path)
    return data


def save_config(config_dict: dict[str, Any], out_path: Path | str) -> Path:
    """Save a configuration dictionary to a YAML file."""
    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    # Strip internal helper keys before saving
    cleaned = {k: v for k, v in config_dict.items() if not k.startswith("_")}
    with open(p_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cleaned, f, sort_keys=False, indent=2, allow_unicode=True)
    return p_out


def extract_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Extract and merge common model parameters with a specific lifecycle section.

    Sections: 'data_generation', 'train', 'eval'
    Returns a unified flat/semi-flat dictionary suitable for configuring components.
    """
    merged: dict[str, Any] = {}

    # 1. Model architecture level parameters
    model_sec = config.get("model", {})
    if isinstance(model_sec, dict):
        for k, v in model_sec.items():
            merged[k] = v
            # Alias model_type as model if needed
            if k == "model_type":
                merged["model"] = v

    # 2. Target lifecycle section
    target_sec = config.get(section, {})
    if isinstance(target_sec, dict):
        for k, v in target_sec.items():
            merged[k] = v

    return merged


def pre_parse_config_path(argv: list[str] | None = None) -> Optional[Path]:
    """Inspect CLI tokens for '--config' or '-c' without triggering full argument parsing.

    Useful for injecting YAML defaults before full argument parsing takes place.
    """
    tokens = sys.argv[1:] if argv is None else argv
    for i, token in enumerate(tokens):
        if token in ("--config", "-c"):
            if i + 1 < len(tokens):
                return Path(tokens[i + 1])
        elif token.startswith("--config="):
            return Path(token.split("=", 1)[1])
        elif token.startswith("-c="):
            return Path(token.split("=", 1)[1])
    return None


def apply_config_to_parser(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    section: str,
) -> argparse.ArgumentParser:
    """Inject configuration values into ArgumentParser action defaults.

    Values in the config act as the new defaults, allowing explicit CLI
    arguments to override them seamlessly.
    """
    merged_cfg = extract_section(config, section)

    # Set parser-level defaults
    parser.set_defaults(**merged_cfg)

    # Update individual action defaults to reflect changes in help messages
    for action in parser._actions:
        dest = action.dest
        if dest in merged_cfg:
            val = merged_cfg[dest]
            # Handle Path conversion if action specifies type=Path
            if action.type == Path and val is not None and not isinstance(val, Path):
                val = Path(val)
            action.default = val

    return parser


__all__ = [
    "load_config",
    "save_config",
    "get_model_config_path",
    "extract_section",
    "pre_parse_config_path",
    "apply_config_to_parser",
    "MODEL_CONFIG_MAP",
    "MODEL_ALIAS_MAP",
]
