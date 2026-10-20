"""
LineFeat: Cross-Image 2D Line Segment Matching Package.

Focused entirely on cross-view matching of pre-detected 2D line segments:
  - MaskLineFeatMatchNet (Default): Continuous Geometry-Field (EDT+AFM) with canonical 255x255 match matrix
  - LineMatcherNet (Baseline): Line-Transformer with Optimal Transport (Sinkhorn)
  - LineMatchPipeline: High-level Python API for pure line matching (defaults to MaskLineFeatMatchNet)
  - Continuous Geometric Fields (EDT + AFM) generator
  - Supervised Line Matching Loss Functions (MaskedMatrixLoss, LineMatchingLoss)
  - Ground-truth generation and PyTorch datasets
"""

from __future__ import annotations

from LineFeat.matcher import LineMatcher, get_distinct_colors
from LineFeat.dataset import LineFeatPairDataset, collate_line_pairs
from LineFeat.model import LineMatcherNet
from LineFeat.loss import LineMatchingLoss, MaskedMatrixLoss
from LineFeat.geometric_field import compute_geometry_field_np, create_6ch_input_tensor
from LineFeat.mask_line_matcher import MaskLineMatcherNet, MaskLineFeatMatchNet
from LineFeat.pipeline import LineMatchPipeline, LineMatchResult
from LineFeat.ground_truth import (
    GTLineMatch,
    check_line_visibility_and_project,
    query_line_semantic,
    extract_ground_truth_line_matches,
)

__all__ = [
    "LineMatcher",
    "LineMatcherNet",
    "MaskLineMatcherNet",
    "MaskLineFeatMatchNet",
    "LineMatchingLoss",
    "MaskedMatrixLoss",
    "compute_geometry_field_np",
    "create_6ch_input_tensor",
    "LineMatchPipeline",
    "LineMatchResult",
    "get_distinct_colors",
    "LineFeatPairDataset",
    "collate_line_pairs",
    "GTLineMatch",
    "check_line_visibility_and_project",
    "query_line_semantic",
    "extract_ground_truth_line_matches",
]

__version__ = "1.0.0"
