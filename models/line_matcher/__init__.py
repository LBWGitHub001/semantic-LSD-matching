"""
Line Matching Models Package (Backward Compatibility Adapter).

Notice: The line matching implementations are canonically organized into:
  - `models.line_transformer`: Standard Line Transformer Matcher & Pipeline
  - `models.mask_line_matcher`: 6-Channel Geometry Field prompted Line Matcher
This package re-exports them for full backward compatibility.
"""

from __future__ import annotations

from models.line_transformer.line_transformer import (
    LineMatcherNet,
    LineGeometryEncoder,
    LineVisualSampler,
    MultiHeadAttentionBlock,
    log_optimal_transport,
)
from models.mask_line_matcher.mask_line_matcher import (
    MaskLineMatcherNet,
    MaskLineFeatMatchNet,
)
from models.line_transformer.loss import LineMatchingLoss, MaskedMatrixLoss
from models.mask_line_matcher.geometric_field import compute_geometry_field_np, create_6ch_input_tensor
from models.line_transformer.matcher import LineMatcher, get_distinct_colors
from models.line_transformer.pipeline import LineMatchPipeline, LineMatchResult

__all__ = [
    "LineMatcherNet",
    "LineGeometryEncoder",
    "LineVisualSampler",
    "MultiHeadAttentionBlock",
    "log_optimal_transport",
    "MaskLineMatcherNet",
    "MaskLineFeatMatchNet",
    "LineMatchingLoss",
    "MaskedMatrixLoss",
    "compute_geometry_field_np",
    "create_6ch_input_tensor",
    "LineMatcher",
    "get_distinct_colors",
    "LineMatchPipeline",
    "LineMatchResult",
]
