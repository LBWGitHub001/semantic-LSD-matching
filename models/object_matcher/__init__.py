"""
Object Matcher Package ("连连看").
"""

from __future__ import annotations

from models.object_matcher.xfeat_object_matcher import (
    XFeatObjectMatcher,
    XFeatBackbone,
    MaskPooling,
    ObjectGeometryEncoder,
    ObjectGraphTransformer,
    ObjectSinkhornMatcher,
)
from models.object_matcher.loss import ObjectMatchingLoss

__all__ = [
    "XFeatObjectMatcher",
    "XFeatBackbone",
    "MaskPooling",
    "ObjectGeometryEncoder",
    "ObjectGraphTransformer",
    "ObjectSinkhornMatcher",
    "ObjectMatchingLoss",
]
