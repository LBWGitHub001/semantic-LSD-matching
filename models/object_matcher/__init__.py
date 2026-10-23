"""
Object Matcher Package ("连连看").
"""

from __future__ import annotations

from models.object_matcher.xfeat_object_matcher import (
    XFeatObjectMatcher,
    XFeatBackbone,
    XFeatPyramidBackbone,
    MaskPooling,
    ObjectGeometryEncoder,
    ObjectGraphTransformer,
    ObjectSinkhornMatcher,
)
from models.object_matcher.xfeat_line_matcher import (
    XFeatLineMatcher,
    PatchLineGeometryEncoder,
    RegionPatchExtractor,
)
from models.object_matcher.loss import ObjectMatchingLoss

__all__ = [
    "XFeatObjectMatcher",
    "XFeatLineMatcher",
    "PatchLineGeometryEncoder",
    "RegionPatchExtractor",
    "XFeatBackbone",
    "XFeatPyramidBackbone",
    "MaskPooling",
    "ObjectGeometryEncoder",
    "ObjectGraphTransformer",
    "ObjectSinkhornMatcher",
    "ObjectMatchingLoss",
]
