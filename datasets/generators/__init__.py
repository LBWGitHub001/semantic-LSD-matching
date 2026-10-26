"""
Dataset Generators Package: Procedural Rooms, Modality Components & Factories.
"""

from __future__ import annotations

from datasets.generators.procedural_room import (
    create_look_at,
    clip_polygon_near_plane,
    compute_polygon_normal,
    apply_shading,
    UnifiedProceduralRoom,
    render_indoor_view,
)

__all__ = [
    "create_look_at",
    "clip_polygon_near_plane",
    "compute_polygon_normal",
    "apply_shading",
    "UnifiedProceduralRoom",
    "render_indoor_view",
]
