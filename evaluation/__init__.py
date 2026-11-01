"""
Evaluation and Benchmark Suite Package.
"""

from __future__ import annotations

from evaluation.checkpoint import resolve_checkpoint_path
from evaluation.model_factory import ModelFactory
from evaluation.dataset_factory import DatasetFactory
from evaluation.evaluators import (
    BaseEvaluator,
    ObjectMatchingEvaluator,
    LineMatchingEvaluator,
    EvaluatorFactory,
    evaluate,
)

__all__ = [
    "resolve_checkpoint_path",
    "ModelFactory",
    "DatasetFactory",
    "BaseEvaluator",
    "ObjectMatchingEvaluator",
    "LineMatchingEvaluator",
    "EvaluatorFactory",
    "evaluate",
]
