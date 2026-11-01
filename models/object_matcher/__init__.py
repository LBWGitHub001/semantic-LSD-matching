"""XFeat Object Matching Networks ("连连看" Bipartite Matcher)."""

from models.object_matcher.xfeat_object_matcher import XFeatObjectMatcher
from models.object_matcher.xfeat_line_matcher import (
    XFeatLineMatcher,
    PatchLineGeometryEncoder,
    RegionPatchExtractor,
)
from models.object_matcher.loss import ObjectMatchingLoss
from models.object_matcher.train import train
from models.object_matcher.eval import evaluate, validate, evaluate_object_pipeline

__all__ = [
    "XFeatObjectMatcher",
    "XFeatLineMatcher",
    "PatchLineGeometryEncoder",
    "RegionPatchExtractor",
    "ObjectMatchingLoss",
    "train",
    "evaluate",
    "validate",
    "evaluate_object_pipeline",
]
