"""
Persistent training-UI hyperparameter configuration.

Stores per-model pinned (常驻) parameters, current values, and a last-train
snapshot under outputs/ui_train_config.json so the console reopens identically.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from web.process_manager import ROOT_DIR

CONFIG_PATH = ROOT_DIR / "outputs" / "ui_train_config.json"

# Parameter definitions shared / specialized by model.
# adjustable=False → shown under 不可调节; still pinnable as read-only 常驻.

_ALWAYS_PINNED = ("data_dir",)

_COMMON_ADJUSTABLE: list[dict[str, Any]] = [
    {
        "id": "data_dir",
        "label": "训练数据集 (Dataset Directory)",
        "type": "dataset",
        "default": "data/xfeat_line_matcher",
        "adjustable": True,
        "always_pinned": True,
    },
    {
        "id": "epochs",
        "label": "总训练轮数 (Epochs)",
        "type": "number",
        "default": 15,
        "min": 1,
        "max": 200,
        "step": 1,
        "adjustable": True,
    },
    {
        "id": "lr",
        "label": "学习率 (Learning Rate)",
        "type": "text",
        "default": "0.0005",
        "adjustable": True,
    },
    {
        "id": "dustbin_weight",
        "label": "垃圾桶损失权重 (Dustbin Weight)",
        "type": "number",
        "default": 0.1,
        "min": 0,
        "max": 2,
        "step": 0.05,
        "adjustable": True,
    },
    {
        "id": "min_area_ratio",
        "label": "最小共视面积门限 (Min Area Ratio)",
        "type": "number",
        "default": 0.40,
        "min": 0,
        "max": 1,
        "step": 0.05,
        "adjustable": True,
    },
    {
        "id": "data_ratio",
        "label": "数据集采样比例 (Data Fraction)",
        "type": "ratio",
        "default": 1.0,
        "min": 0.01,
        "max": 1.0,
        "step": 0.01,
        "adjustable": True,
        "hint": "随机抽取数据集内容的比例（如 0.5 表示使用一半数据）",
    },
    {
        "id": "label",
        "label": "自定义训练标签 (Optional Label)",
        "type": "text",
        "default": "",
        "placeholder": "例如: lr0005_ablation, baseline_cad",
        "adjustable": True,
        "hint": "写入复盘标识与 Checkpoint 元数据",
    },
    {
        "id": "pyramid",
        "label": "多尺度金字塔 (Pyramid FPN)",
        "type": "checkbox",
        "default": True,
        "adjustable": True,
    },
    {
        "id": "amp",
        "label": "混合精度加速 (AMP FP16)",
        "type": "checkbox",
        "default": True,
        "adjustable": True,
    },
    {
        "id": "resume",
        "label": "导入模型 / 断点续训 (Resume)",
        "type": "resume",
        "default": "",
        "adjustable": True,
        "hint": "选择已有 .pth，训练时以 --resume 续训",
    },
    {
        "id": "d_model",
        "label": "特征维度 d_model",
        "type": "number",
        "default": 128,
        "min": 32,
        "max": 512,
        "step": 32,
        "adjustable": True,
    },
    {
        "id": "num_layers",
        "label": "Transformer 层数 (num_layers)",
        "type": "number",
        "default": 2,
        "min": 1,
        "max": 8,
        "step": 1,
        "adjustable": True,
    },
    {
        "id": "nhead",
        "label": "注意力头数 (nhead)",
        "type": "number",
        "default": 4,
        "min": 1,
        "max": 16,
        "step": 1,
        "adjustable": True,
    },
    {
        "id": "weight_decay",
        "label": "权重衰减 (weight_decay)",
        "type": "number",
        "default": 0.0001,
        "min": 0,
        "max": 0.1,
        "step": 0.0001,
        "adjustable": True,
    },
    {
        "id": "num_workers",
        "label": "DataLoader workers",
        "type": "number",
        "default": 4,
        "min": 0,
        "max": 16,
        "step": 1,
        "adjustable": True,
    },
    {
        "id": "save_interval",
        "label": "历史 checkpoint 间隔 (save_interval)",
        "type": "number",
        "default": 5,
        "min": 1,
        "max": 50,
        "step": 1,
        "adjustable": True,
    },
    {
        "id": "val_interval",
        "label": "验证间隔 epochs (val_interval)",
        "type": "number",
        "default": 1,
        "min": 1,
        "max": 20,
        "step": 1,
        "adjustable": True,
    },
]

_COMMON_FIXED: list[dict[str, Any]] = [
    {
        "id": "batch_size",
        "label": "Batch Size",
        "type": "text",
        "default": "1",
        "adjustable": False,
        "hint": "当前训练管线固定为 1（按 pair collate）",
    },
    {
        "id": "sinkhorn_iters",
        "label": "Sinkhorn 迭代次数",
        "type": "text",
        "default": "20",
        "adjustable": False,
    },
    {
        "id": "feat_dim",
        "label": "骨干特征维 feat_dim",
        "type": "text",
        "default": "64",
        "adjustable": False,
    },
    {
        "id": "device",
        "label": "训练设备 (device)",
        "type": "text",
        "default": "auto (cuda if available)",
        "adjustable": False,
    },
]

_MOTION_PARAM = {
    "id": "motion_weight",
    "label": "位移场损失权重 (Motion Weight)",
    "type": "number",
    "default": 0.4,
    "min": 0,
    "max": 2,
    "step": 0.05,
    "adjustable": True,
}

_MOTION_MARGIN = {
    "id": "motion_margin",
    "label": "位移容差 (Motion Margin)",
    "type": "number",
    "default": 0.15,
    "min": 0,
    "max": 1,
    "step": 0.05,
    "adjustable": True,
}

_SEMANTIC_PARAMS: list[dict[str, Any]] = [
    {
        "id": "region_weight",
        "label": "区域 KL 权重 (Region Weight)",
        "type": "number",
        "default": 1.0,
        "min": 0,
        "max": 5,
        "step": 0.05,
        "adjustable": True,
    },
    {
        "id": "line_weight",
        "label": "直线 KL 权重 (Line Weight)",
        "type": "number",
        "default": 1.0,
        "min": 0,
        "max": 5,
        "step": 0.05,
        "adjustable": True,
    },
    {
        "id": "sem_weight",
        "label": "语义集中 JSD 权重 (Sem Weight)",
        "type": "number",
        "default": 0.1,
        "min": 0,
        "max": 2,
        "step": 0.05,
        "adjustable": True,
    },
    {
        "id": "geo_weight",
        "label": "邻接保持权重 (Geo Weight)",
        "type": "number",
        "default": 0.2,
        "min": 0,
        "max": 2,
        "step": 0.05,
        "adjustable": True,
    },
    {
        "id": "adj_sigma",
        "label": "邻接温度 σ (adj_sigma)",
        "type": "number",
        "default": 0.3,
        "min": 0.05,
        "max": 2,
        "step": 0.05,
        "adjustable": True,
    },
    {
        "id": "line_sample_points",
        "label": "沿线采样点数 T",
        "type": "number",
        "default": 8,
        "min": 2,
        "max": 32,
        "step": 1,
        "adjustable": True,
    },
]

_DEFAULT_PINNED_OBJECT = [
    "data_dir",
    "epochs",
    "lr",
    "dustbin_weight",
    "motion_weight",
    "min_area_ratio",
    "data_ratio",
    "label",
    "pyramid",
    "amp",
    "resume",
]

_DEFAULT_PINNED_SEMANTIC = [
    "data_dir",
    "epochs",
    "lr",
    "dustbin_weight",
    "region_weight",
    "line_weight",
    "sem_weight",
    "geo_weight",
    "min_area_ratio",
    "data_ratio",
    "label",
    "pyramid",
    "amp",
    "resume",
]

_DEFAULT_PINNED_LINE = [
    "data_dir",
    "epochs",
    "lr",
    "min_matches",
    "d_model",
    "num_layers",
    "label",
    "amp",
    "resume",
]


def _params_for_object_family(*, with_motion: bool, semantic: bool) -> list[dict[str, Any]]:
    params = deepcopy(_COMMON_ADJUSTABLE)
    if semantic:
        # motion disabled for semantic_line; insert semantic losses after dustbin
        insert_at = next(i for i, p in enumerate(params) if p["id"] == "min_area_ratio")
        for offset, p in enumerate(_SEMANTIC_PARAMS):
            params.insert(insert_at + offset, deepcopy(p))
        params.append(
            {
                "id": "motion_weight",
                "label": "位移场损失权重 (Motion Weight)",
                "type": "number",
                "default": 0.0,
                "min": 0,
                "max": 0,
                "step": 0.05,
                "adjustable": False,
                "hint": "semantic_line_matcher 默认关闭旧 motion（固定 0）",
            }
        )
    elif with_motion:
        insert_at = next(i for i, p in enumerate(params) if p["id"] == "min_area_ratio")
        params.insert(insert_at, deepcopy(_MOTION_PARAM))
        params.insert(insert_at + 1, deepcopy(_MOTION_MARGIN))
    params.extend(deepcopy(_COMMON_FIXED))
    return params


def _params_for_line_family() -> list[dict[str, Any]]:
    params = [
        {
            "id": "data_dir",
            "label": "训练数据集 (Dataset Directory)",
            "type": "dataset",
            "default": "data/xfeat_line_matcher",
            "adjustable": True,
        },
        {
            "id": "epochs",
            "label": "总训练轮数 (Epochs)",
            "type": "number",
            "default": 12,
            "min": 1,
            "max": 200,
            "step": 1,
            "adjustable": True,
        },
        {
            "id": "lr",
            "label": "学习率 (Learning Rate)",
            "type": "text",
            "default": "0.0001",
            "adjustable": True,
        },
        {
            "id": "min_matches",
            "label": "最少 GT 匹配数 (min_matches)",
            "type": "number",
            "default": 3,
            "min": 1,
            "max": 50,
            "step": 1,
            "adjustable": True,
        },
        {
            "id": "d_model",
            "label": "特征维度 d_model",
            "type": "number",
            "default": 128,
            "min": 32,
            "max": 512,
            "step": 32,
            "adjustable": True,
        },
        {
            "id": "num_layers",
            "label": "Transformer 层数",
            "type": "number",
            "default": 2,
            "min": 1,
            "max": 8,
            "step": 1,
            "adjustable": True,
        },
        {
            "id": "nhead",
            "label": "注意力头数",
            "type": "number",
            "default": 4,
            "min": 1,
            "max": 16,
            "step": 1,
            "adjustable": True,
        },
        {
            "id": "label",
            "label": "自定义训练标签",
            "type": "text",
            "default": "",
            "adjustable": True,
        },
        {
            "id": "amp",
            "label": "混合精度 AMP FP16",
            "type": "checkbox",
            "default": True,
            "adjustable": True,
        },
        {
            "id": "resume",
            "label": "导入模型 / 断点续训",
            "type": "resume",
            "default": "",
            "adjustable": True,
        },
        {
            "id": "num_workers",
            "label": "DataLoader workers",
            "type": "number",
            "default": 4,
            "min": 0,
            "max": 16,
            "step": 1,
            "adjustable": True,
        },
    ]
    params.extend(deepcopy(_COMMON_FIXED))
    return params


def _schema(title: str, family: str, params: list[dict[str, Any]], pinned: list[str]) -> dict[str, Any]:
    return {
        "title": title,
        "family": family,
        "params": params,
        "default_pinned": list(pinned),
    }


MODEL_SCHEMAS: dict[str, dict[str, Any]] = {
    name: _schema(title, family, params, pinned)
    for names, title, family, params, pinned in (
        (
            ("semantic_line_matcher", "semantic_line"),
            "SemanticLineMatcher",
            "object",
            _params_for_object_family(with_motion=False, semantic=True),
            _DEFAULT_PINNED_SEMANTIC,
        ),
        (
            ("xfeat_line_matcher", "xfeat_line"),
            "XFeatLineMatcher",
            "object",
            _params_for_object_family(with_motion=True, semantic=False),
            _DEFAULT_PINNED_OBJECT,
        ),
        (
            ("xfeat_object", "xfeat"),
            "XFeatObjectMatcher",
            "object",
            _params_for_object_family(with_motion=True, semantic=False),
            _DEFAULT_PINNED_OBJECT,
        ),
        (
            ("mask_line", "mask"),
            "MaskLineFeatMatchNet",
            "line",
            _params_for_line_family(),
            _DEFAULT_PINNED_LINE,
        ),
        (
            ("line_transformer", "linefeat"),
            "LineMatcherNet",
            "line",
            _params_for_line_family(),
            _DEFAULT_PINNED_LINE,
        ),
    )
    for name in names
}


def _default_values(params: list[dict[str, Any]]) -> dict[str, Any]:
    return {p["id"]: deepcopy(p["default"]) for p in params}


def _ensure_always_pinned(pinned: list[str], valid_ids: set[str]) -> list[str]:
    result = [x for x in _ALWAYS_PINNED if x in valid_ids]
    for item in pinned:
        if item in valid_ids and item not in result:
            result.append(item)
    return result


def _default_model_state(model: str) -> dict[str, Any]:
    schema = MODEL_SCHEMAS.get(model) or MODEL_SCHEMAS["xfeat_line_matcher"]
    valid_ids = {p["id"] for p in schema["params"]}
    return {
        "pinned": _ensure_always_pinned(list(schema["default_pinned"]), valid_ids),
        "values": _default_values(schema["params"]),
    }


def default_config() -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name in MODEL_SCHEMAS:
        models[name] = _default_model_state(name)
    return {
        "version": 1,
        "active_model": "xfeat_line_matcher",
        "models": models,
        "last_train_snapshot": None,
    }


def load_config() -> dict[str, Any]:
    cfg = default_config()
    if not CONFIG_PATH.is_file():
        return cfg
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            disk = json.load(f)
    except Exception:
        return cfg

    if not isinstance(disk, dict):
        return cfg

    cfg["active_model"] = disk.get("active_model") or cfg["active_model"]
    if cfg["active_model"] not in MODEL_SCHEMAS:
        cfg["active_model"] = "xfeat_line_matcher"

    disk_models = disk.get("models") or {}
    for name, schema in MODEL_SCHEMAS.items():
        base = _default_model_state(name)
        saved = disk_models.get(name) or {}
        pinned = saved.get("pinned")
        if isinstance(pinned, list):
            valid_ids = {p["id"] for p in schema["params"]}
            base["pinned"] = _ensure_always_pinned(pinned, valid_ids)
        values = saved.get("values") or {}
        if isinstance(values, dict):
            for k, v in values.items():
                if k in base["values"]:
                    base["values"][k] = v
        cfg["models"][name] = base

    if disk.get("last_train_snapshot") is not None:
        cfg["last_train_snapshot"] = disk["last_train_snapshot"]
    return cfg


def save_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge incoming partial/full config onto disk and write."""
    current = load_config()
    if not isinstance(cfg, dict):
        return current

    if "active_model" in cfg and cfg["active_model"] in MODEL_SCHEMAS:
        current["active_model"] = cfg["active_model"]

    incoming_models = cfg.get("models") or {}
    if isinstance(incoming_models, dict):
        for name, state in incoming_models.items():
            if name not in MODEL_SCHEMAS or not isinstance(state, dict):
                continue
            slot = current["models"].setdefault(name, _default_model_state(name))
            if isinstance(state.get("pinned"), list):
                valid_ids = {p["id"] for p in MODEL_SCHEMAS[name]["params"]}
                slot["pinned"] = _ensure_always_pinned(state["pinned"], valid_ids)
            if isinstance(state.get("values"), dict):
                for k, v in state["values"].items():
                    if k in slot["values"]:
                        slot["values"][k] = v

    if "last_train_snapshot" in cfg:
        current["last_train_snapshot"] = cfg["last_train_snapshot"]

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    return current


def build_api_payload(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    schemas = {
        name: {
            "title": meta["title"],
            "family": meta["family"],
            "params": meta["params"],
            "default_pinned": meta["default_pinned"],
        }
        for name, meta in MODEL_SCHEMAS.items()
    }
    return {
        "config_path": str(CONFIG_PATH.relative_to(ROOT_DIR)),
        "schemas": schemas,
        "config": cfg,
    }
