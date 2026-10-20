"""
LineFeat: Semantic & Geometric 2D Line Segment Matching Package.

Consolidated modules for:
  - 2D Line Segment Detection (LSD)
  - Deep Learning Line Descriptor Extraction
  - Feature Matching with MNN, Lowe's Ratio Test, & Semantic Gating
  - Ground-Truth Generation from 3D Wireframes & Depth Occlusion
  - PyTorch Dataset for End-to-End Training & Evaluation
"""

from __future__ import annotations

from LineFeat.detector import LineDetector
from LineFeat.descriptor import LineDescriptor
from LineFeat.matcher import LineMatcher, get_distinct_colors
from LineFeat.dataset import LineFeatPairDataset, collate_line_pairs
from LineFeat.model import LineMatcherNet
from LineFeat.loss import LineMatchingLoss, MaskedMatrixLoss
from LineFeat.geometric_field import compute_geometry_field_np, create_6ch_input_tensor
from LineFeat.mask_line_matcher import MaskLineMatcherNet
from LineFeat.neighborhood import (
    LineNeighborhoodTransformer,
    compute_strip_zncc,
    extract_deep_strip_descriptor,
)
from LineFeat.pipeline import LineMatchPipeline, LineMatchResult
from LineFeat.ground_truth import (
    GTLineMatch,
    check_line_visibility_and_project,
    query_line_semantic,
    extract_ground_truth_line_matches,
)

__all__ = [
    "LineDetector",
    "LineDescriptor",
    "LineMatcher",
    "LineMatcherNet",
    "LineMatchingLoss",
    "MaskedMatrixLoss",
    "MaskLineMatcherNet",
    "compute_geometry_field_np",
    "create_6ch_input_tensor",
    "LineMatchPipeline",
    "LineMatchResult",
    "get_distinct_colors",
    "LineNeighborhoodTransformer",
    "compute_strip_zncc",
    "extract_deep_strip_descriptor",
    "LineFeatPairDataset",
    "collate_line_pairs",
    "GTLineMatch",
    "check_line_visibility_and_project",
    "query_line_semantic",
    "extract_ground_truth_line_matches",
]

__version__ = "0.1.0"
