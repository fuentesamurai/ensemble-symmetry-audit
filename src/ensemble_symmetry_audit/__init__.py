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
from .soft_api import soft_audit
from .soft_detectors import (
    soft_balanced_input_symmetry,
    soft_continuity,
    soft_monotonicity,
    soft_pareto_unanimity,
    soft_permutation_invariance,
    soft_regime_flip_invariance,
)
from .hypothesis_search import (
    shrink_hard_counterexample,
    shrink_soft_counterexample,
)
from . import strategies

__all__ = [
    # core
    "audit",
    "AuditReport",
    "DetectorResult",
    # hard detectors
    "balanced_input_symmetry",
    "regime_flip_invariance",
    "null_majority_abstention",
    "monotonicity",
    "permutation_invariance",
    "tie_break_determinism",
    "pareto_unanimity",
    "independence_of_irrelevant_alternatives",
    # soft
    "soft_audit",
    "soft_balanced_input_symmetry",
    "soft_continuity",
    "soft_monotonicity",
    "soft_pareto_unanimity",
    "soft_permutation_invariance",
    "soft_regime_flip_invariance",
    # hypothesis adversarial
    "shrink_hard_counterexample",
    "shrink_soft_counterexample",
    "strategies",
]

__version__ = "0.3.0"
