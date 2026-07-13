"""Diff-In: Differential Influence estimator.

Reference: Tan et al., "Understanding Data Influence with Differential
Approximation" (arXiv:2508.14648).
"""

from .estimator import CheckpointSpec, DiffInEstimator, InfluenceResult
from . import utils

__all__ = [
    "CheckpointSpec",
    "DiffInEstimator",
    "InfluenceResult",
    "utils",
]
