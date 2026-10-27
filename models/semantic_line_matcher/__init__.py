"""
SemanticLineMatcher: dual region + line matching under semantic regions.

Line descriptors = along-line texture features concatenated with parent region features.
Outputs region Sinkhorn (P_R) and line Sinkhorn (P_L) in one forward pass.
"""

from __future__ import annotations

from models.semantic_line_matcher.loss import SemanticLineMatchingLoss
from models.semantic_line_matcher.model import SemanticLineMatcher

__all__ = [
    "SemanticLineMatcher",
    "SemanticLineMatchingLoss",
]
