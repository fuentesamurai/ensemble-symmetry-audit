"""Property-based bias detectors for voting ensembles."""

from .api import audit, AuditReport
from .detectors import (
    DetectorResult,
    balanced_input_symmetry,
    regime_flip_invariance,
    null_majority_abstention,
    monotonicity,
    permutation_invariance,
    tie_break_determinism,
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
]

__version__ = "0.1.0"
