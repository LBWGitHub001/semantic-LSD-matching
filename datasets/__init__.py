"""
Unified Datasets Package for Line and Object Matching.
"""

from __future__ import annotations

from datasets.object_dataset import (
    ObjectMatchingDataset,
    collate_single_pair,
    predict_bipartite_matches,
    visualize_object_matches,
)
from datasets.line_dataset import (
    LineFeatPairDataset,
    collate_line_pairs,
)
from datasets.read_structured3d import (
    STRUCTURED3D_CLASSES,
    Camera,
    Structured3DDataset,
    Structured3DScene,
    Structured3DView,
)

__all__ = [
    "ObjectMatchingDataset",
    "collate_single_pair",
    "predict_bipartite_matches",
    "visualize_object_matches",
    "LineFeatPairDataset",
    "collate_line_pairs",
    "STRUCTURED3D_CLASSES",
    "Camera",
    "Structured3DDataset",
    "Structured3DScene",
    "Structured3DView",
]
