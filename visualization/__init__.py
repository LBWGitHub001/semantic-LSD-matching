"""
Unified Visualization Suite for Semantic LSD Matching.
Provides:
1. MatchingVisualizer: High-performance dual-row semantic matching canvas and bipartite object matching (OpenCV/NumPy only).
2. ArchitectureVisualizer: 4K modular network topology with explicit tensor dimensions (Lazy loaded).
3. CurveVisualizer: Trajectory and cross-run comparison metric plotting (Lazy loaded).
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

# Core matching visualizer uses only OpenCV + NumPy (Zero heavy dependencies)
from visualization.matching_visualizer import (
    DISTINCT_COLORS,
    blend_mask,
    draw_dashed_line,
    draw_mask_boundary,
    draw_text_with_pill,
    save_pair_visualization,
    visualize_line_matches,
    visualize_object_matches,
    visualize_semantic_region_matches,
)


def render_model_architecture(model_name: str, out_path: Optional[Path | str] = None, dpi: int = 180) -> Path:
    """Lazy-loaded wrapper for architecture diagram rendering."""
    from visualization.architecture_visualizer import render_model_architecture as _fn
    return _fn(model_name=model_name, out_path=out_path, dpi=dpi)


def render_all_models(output_dir: Optional[Path | str] = None) -> dict[str, Path]:
    """Lazy-loaded wrapper for batch rendering all model architecture diagrams."""
    from visualization.architecture_visualizer import render_all_models as _fn
    return _fn(output_dir=output_dir)


def plot_training_trajectory(
    history: list[dict[str, Any]],
    out_path: Path | str,
    title: str = "Training Trajectory",
    dpi: int = 150,
) -> Path:
    """Lazy-loaded wrapper for training trajectory curve plotting."""
    from visualization.curve_visualizer import plot_training_trajectory as _fn
    return _fn(history=history, out_path=out_path, title=title, dpi=dpi)


__all__ = [
    # Core matching & diagnosis (Lightweight)
    "DISTINCT_COLORS",
    "blend_mask",
    "draw_mask_boundary",
    "draw_dashed_line",
    "draw_text_with_pill",
    "visualize_semantic_region_matches",
    "save_pair_visualization",
    "visualize_line_matches",
    "visualize_object_matches",
    # Architecture & Curve functions (Lazy)
    "render_model_architecture",
    "render_all_models",
    "plot_training_trajectory",
]
