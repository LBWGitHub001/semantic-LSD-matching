"""
Baseline Line-Transformer Matcher (LineMatcherNet).
"""

from __future__ import annotations

from models.line_transformer.line_transformer import (
    LineMatcherNet,
    LineGeometryEncoder,
    LineVisualSampler,
    MultiHeadAttentionBlock,
    log_optimal_transport,
)
from models.line_transformer.loss import LineMatchingLoss, MaskedMatrixLoss
from models.line_transformer.matcher import LineMatcher, get_distinct_colors
from models.line_transformer.pipeline import LineMatchPipeline, LineMatchResult

__all__ = [
    "LineMatcherNet",
    "LineGeometryEncoder",
    "LineVisualSampler",
    "MultiHeadAttentionBlock",
    "log_optimal_transport",
    "LineMatchingLoss",
    "MaskedMatrixLoss",
    "LineMatcher",
    "get_distinct_colors",
    "LineMatchPipeline",
    "LineMatchResult",
]
