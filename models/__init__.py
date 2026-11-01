"""
Unified Model Registry and Factory for the Matching Networks:
1. Object Matching Network: XFeatObjectMatcher & XFeatLineMatcher (models/object_matcher)
2. 6-Channel Geometry Field Line Matcher: MaskLineFeatMatchNet (models/mask_line_matcher)
3. SuperPoint: Interest points, descriptors, and line endpoints (models/superpoint)
"""

from __future__ import annotations

import torch.nn as nn

# 1. Object-Level Matchers
from models.object_matcher import (
    XFeatObjectMatcher,
    XFeatLineMatcher,
    PatchLineGeometryEncoder,
    RegionPatchExtractor,
    ObjectMatchingLoss,
)

# 2. 6-Channel Geometry Field Line Matcher
from models.mask_line_matcher import (
    MaskLineFeatMatchNet,
    MaskLineMatcherNet,
    LineMatchingLoss,
    MaskedMatrixLoss,
)

# 3. SuperPoint interest points, descriptors, and line endpoints
from models.superpoint import (
    SuperPoint,
    SuperPointLoss,
    SuperPointLineLoss,
    SuperPointDescriptorLoss,
    SuperPointDetectorLoss,
)

MODEL_REGISTRY = {
    # Model 1: Object Bipartite Matchers
    "xfeat_object": XFeatObjectMatcher,
    "xfeat": XFeatObjectMatcher,
    "object": XFeatObjectMatcher,
    # Model 1.5: Multi-Modal Object + Straight Line Hybrid Matcher
    "xfeat_line_matcher": XFeatLineMatcher,
    "xfeat_line": XFeatLineMatcher,
    "line_object": XFeatLineMatcher,
    # Model 2: 6-Channel Continuous Geometry Field Line Matcher
    "mask_line": MaskLineFeatMatchNet,
    "mask": MaskLineFeatMatchNet,
    # Model 3: SuperPoint
    "SuperPoint": SuperPoint,
    "superpoint": SuperPoint,
}

LOSS_REGISTRY = {
    "xfeat_object": ObjectMatchingLoss,
    "xfeat": ObjectMatchingLoss,
    "object": ObjectMatchingLoss,
    "xfeat_line_matcher": ObjectMatchingLoss,
    "xfeat_line": ObjectMatchingLoss,
    "line_object": ObjectMatchingLoss,
    "mask_line": LineMatchingLoss,
    "mask": LineMatchingLoss,
    "SuperPoint": SuperPointLoss,
    "superpoint": SuperPointLoss,
}

CANONICAL_MODEL_NAMES = {
    "xfeat_object": "xfeat_object",
    "xfeat": "xfeat_object",
    "object": "xfeat_object",
    "xfeat_line_matcher": "xfeat_line_matcher",
    "xfeat_line": "xfeat_line_matcher",
    "line_object": "xfeat_line_matcher",
    "mask_line": "mask_line",
    "mask": "mask_line",
    "SuperPoint": "SuperPoint",
    "superpoint": "SuperPoint",
}

OBJECT_FAMILY_MODELS = {"xfeat_object", "xfeat_line_matcher"}
LINE_FEATURE_MODELS = {"xfeat_line_matcher"}
POINT_MODELS = {"SuperPoint"}


def canonical_model_name(model_name: str) -> str:
    """Resolve any registered alias to its canonical model name."""
    raw = model_name or ""
    for candidate in (raw, raw.lower()):
        if candidate in CANONICAL_MODEL_NAMES:
            return CANONICAL_MODEL_NAMES[candidate]
    raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")


def model_family(model_name: str) -> str:
    """Return 'object', 'point', or 'line'."""
    name = canonical_model_name(model_name)
    if name in POINT_MODELS:
        return "point"
    if name in OBJECT_FAMILY_MODELS:
        return "object"
    return "line"


def uses_patch_lines(model_name: str) -> bool:
    """Whether the model consumes per-region 5D patch lines."""
    return canonical_model_name(model_name) in LINE_FEATURE_MODELS


def build_model(model_name: str, **kwargs) -> nn.Module:
    """Builds a model instance by name."""
    name = canonical_model_name(model_name)
    return MODEL_REGISTRY[name](**kwargs)


def build_loss(model_name: str, **kwargs) -> nn.Module:
    """Builds the corresponding loss function for a model."""
    name = canonical_model_name(model_name)
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss for model '{model_name}'. Available: {list(LOSS_REGISTRY.keys())}")
    if name == "SuperPoint":
        stage = kwargs.pop("stage", None)
        if stage == "descriptor":
            return SuperPointDescriptorLoss(**kwargs)
        elif stage == "detector":
            return SuperPointDetectorLoss(**kwargs)
        return SuperPointLoss(**kwargs)
    return LOSS_REGISTRY[name](**kwargs)


def get_available_models() -> list[str]:
    """Returns list of supported model aliases."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "build_model",
    "build_loss",
    "get_available_models",
    "canonical_model_name",
    "model_family",
    "uses_patch_lines",
    "MODEL_REGISTRY",
    "LOSS_REGISTRY",
    "XFeatObjectMatcher",
    "XFeatLineMatcher",
    "PatchLineGeometryEncoder",
    "RegionPatchExtractor",
    "ObjectMatchingLoss",
    "MaskLineFeatMatchNet",
    "MaskLineMatcherNet",
    "LineMatchingLoss",
    "MaskedMatrixLoss",
    "SuperPoint",
    "SuperPointLoss",
    "SuperPointLineLoss",
    "SuperPointDescriptorLoss",
    "SuperPointDetectorLoss",
]
