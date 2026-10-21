"""
Unified Model Registry and Factory for Object & Line Matching Networks.
"""

from __future__ import annotations

import torch.nn as nn

from models.object_matcher import (
    XFeatObjectMatcher,
    ObjectMatchingLoss,
)
from models.line_matcher import (
    MaskLineFeatMatchNet,
    MaskLineMatcherNet,
    LineMatcherNet,
    LineMatchingLoss,
    MaskedMatrixLoss,
)

MODEL_REGISTRY = {
    # Object Bipartite Matchers ("连连看")
    "xfeat_object": XFeatObjectMatcher,
    "xfeat": XFeatObjectMatcher,
    "object": XFeatObjectMatcher,
    # Line Matchers (6-Channel Geometry Field)
    "mask_line": MaskLineFeatMatchNet,
    "mask": MaskLineFeatMatchNet,
    # Line Matchers (Baseline Line-Transformer)
    "line_transformer": LineMatcherNet,
    "linefeat": LineMatcherNet,
}

LOSS_REGISTRY = {
    "xfeat_object": ObjectMatchingLoss,
    "xfeat": ObjectMatchingLoss,
    "object": ObjectMatchingLoss,
    "mask_line": LineMatchingLoss,
    "mask": LineMatchingLoss,
    "line_transformer": LineMatchingLoss,
    "linefeat": LineMatchingLoss,
}


def build_model(model_name: str, **kwargs) -> nn.Module:
    """Builds a model instance by name."""
    name = model_name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[name]
    return model_cls(**kwargs)


def build_loss(model_name: str, **kwargs) -> nn.Module:
    """Builds the corresponding loss function for a model."""
    name = model_name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss for model '{model_name}'. Available: {list(LOSS_REGISTRY.keys())}")
    loss_cls = LOSS_REGISTRY[name]
    return loss_cls(**kwargs)


def get_available_models() -> list[str]:
    """Returns list of supported model aliases."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "build_model",
    "build_loss",
    "get_available_models",
    "MODEL_REGISTRY",
    "LOSS_REGISTRY",
    "XFeatObjectMatcher",
    "ObjectMatchingLoss",
    "MaskLineFeatMatchNet",
    "MaskLineMatcherNet",
    "LineMatcherNet",
    "LineMatchingLoss",
    "MaskedMatrixLoss",
]
