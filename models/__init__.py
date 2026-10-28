"""
Unified Model Registry and Factory for the Three Matching Networks:
1. Object Matching Network ("连连看"): XFeatObjectMatcher (models/object_matcher)
2. 6-Channel Geometry Field Line Matcher: MaskLineFeatMatchNet (models/mask_line_matcher)
3. Baseline Line Transformer Matcher: LineMatcherNet (models/line_transformer)
"""

from __future__ import annotations

import torch.nn as nn

# 1. Object-Level Matcher ("连连看")
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
)

# 3. Baseline Line Transformer Matcher
from models.line_transformer import (
    LineMatcherNet,
    LineMatchingLoss,
    MaskedMatrixLoss,
    LineMatcher,
    LineMatchPipeline,
)

# 4. Semantic region-conditioned dual line matcher
from models.semantic_line_matcher import (
    SemanticLineMatcher,
    SemanticLineMatchingLoss,
)

# 5. DeepLSD field homography
from models.homo_matcher import (
    HomoMatcher,
    HomoMatcherLoss,
)

MODEL_REGISTRY = {
    # Model 1: Object Bipartite Matchers ("连连看")
    "xfeat_object": XFeatObjectMatcher,
    "xfeat": XFeatObjectMatcher,
    "object": XFeatObjectMatcher,
    # Model 1.5: Multi-Modal Object + Straight Line Hybrid Matcher
    "xfeat_line_matcher": XFeatLineMatcher,
    "xfeat_line": XFeatLineMatcher,
    "line_object": XFeatLineMatcher,
    # Model 1.6: Dual region + texture-line matcher
    "semantic_line_matcher": SemanticLineMatcher,
    "semantic_line": SemanticLineMatcher,
    # Model 2: 6-Channel Continuous Geometry Field Line Matcher
    "mask_line": MaskLineFeatMatchNet,
    "mask": MaskLineFeatMatchNet,
    # Model 3: Baseline Line-Transformer Matcher
    "line_transformer": LineMatcherNet,
    "linefeat": LineMatcherNet,
    "homo_matcher": HomoMatcher,
}

LOSS_REGISTRY = {
    "xfeat_object": ObjectMatchingLoss,
    "xfeat": ObjectMatchingLoss,
    "object": ObjectMatchingLoss,
    "xfeat_line_matcher": ObjectMatchingLoss,
    "xfeat_line": ObjectMatchingLoss,
    "line_object": ObjectMatchingLoss,
    "semantic_line_matcher": SemanticLineMatchingLoss,
    "semantic_line": SemanticLineMatchingLoss,
    "mask_line": LineMatchingLoss,
    "mask": LineMatchingLoss,
    "line_transformer": LineMatchingLoss,
    "linefeat": LineMatchingLoss,
    "homo_matcher": HomoMatcherLoss,
}

CANONICAL_MODEL_NAMES = {
    "xfeat_object": "xfeat_object",
    "xfeat": "xfeat_object",
    "object": "xfeat_object",
    "xfeat_line_matcher": "xfeat_line_matcher",
    "xfeat_line": "xfeat_line_matcher",
    "line_object": "xfeat_line_matcher",
    "semantic_line_matcher": "semantic_line_matcher",
    "semantic_line": "semantic_line_matcher",
    "mask_line": "mask_line",
    "mask": "mask_line",
    "line_transformer": "line_transformer",
    "linefeat": "line_transformer",
    "homo_matcher": "homo_matcher",
}

OBJECT_FAMILY_MODELS = {"xfeat_object", "xfeat_line_matcher", "semantic_line_matcher"}
LINE_FEATURE_MODELS = {"xfeat_line_matcher", "semantic_line_matcher"}
HOMOGRAPHY_MODELS = {"homo_matcher"}


def canonical_model_name(model_name: str) -> str:
    """Resolve any registered alias to its canonical model name."""
    name = (model_name or "").lower()
    if name not in CANONICAL_MODEL_NAMES:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return CANONICAL_MODEL_NAMES[name]


def model_family(model_name: str) -> str:
    """Return 'object', 'homography', or 'line'."""
    name = canonical_model_name(model_name)
    if name in HOMOGRAPHY_MODELS:
        return "homography"
    if name in OBJECT_FAMILY_MODELS:
        return "object"
    return "line"


def uses_patch_lines(model_name: str) -> bool:
    """Whether the model consumes per-region 5D patch lines."""
    return canonical_model_name(model_name) in LINE_FEATURE_MODELS


def is_semantic_line_model(model_name: str) -> bool:
    """Whether the model is the dual region + line Sinkhorn matcher."""
    return canonical_model_name(model_name) == "semantic_line_matcher"


def is_homography_model(model_name: str) -> bool:
    """Whether the model predicts a pair of homographies from DeepLSD fields."""
    return canonical_model_name(model_name) in HOMOGRAPHY_MODELS


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
    "canonical_model_name",
    "model_family",
    "uses_patch_lines",
    "is_semantic_line_model",
    "is_homography_model",
    "MODEL_REGISTRY",
    "LOSS_REGISTRY",
    "XFeatObjectMatcher",
    "XFeatLineMatcher",
    "PatchLineGeometryEncoder",
    "RegionPatchExtractor",
    "ObjectMatchingLoss",
    "MaskLineFeatMatchNet",
    "MaskLineMatcherNet",
    "LineMatcherNet",
    "LineMatchingLoss",
    "MaskedMatrixLoss",
    "LineMatcher",
    "LineMatchPipeline",
    "SemanticLineMatcher",
    "SemanticLineMatchingLoss",
    "HomoMatcher",
    "HomoMatcherLoss",
]
