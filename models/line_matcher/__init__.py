"""
Line Matching Models Package.
"""

from __future__ import annotations

from models.line_matcher.line_transformer import (
    LineMatcherNet,
    LineGeometryEncoder,
    LineVisualSampler,
    MultiHeadAttentionBlock,
    log_optimal_transport,
)
from models.line_matcher.mask_line_matcher import (
    MaskLineMatcherNet,
    MaskLineFeatMatchNet,
)
from models.line_matcher.loss import LineMatchingLoss, MaskedMatrixLoss
from models.line_matcher.geometric_field import compute_geometry_field_np, create_6ch_input_tensor
from models.line_matcher.matcher import LineMatcher, get_distinct_colors
from models.line_matcher.pipeline import LineMatchPipeline, LineMatchResult

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
