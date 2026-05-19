"""Property-based audit of voting ensemble aggregators."""

from .api import audit, AuditReport
from .detectors import (
    DetectorResult,
    balanced_input_symmetry,
    regime_flip_invariance,
    null_majority_abstention,
    monotonicity,
    permutation_invariance,
    tie_break_determinism,
    pareto_unanimity,
    independence_of_irrelevant_alternatives,
)

__all__ = [
    "audit",
    "AuditReport",
    "DetectorResult",
    "balanced_input_symmetry",
    "regime_flip_invariance",
    "null_majority_abstention",
    "monotonicity",
    "permutation_invariance",
    "tie_break_determinism",
    "pareto_unanimity",
    "independence_of_irrelevant_alternatives",
]

__version__ = "0.2.0"
