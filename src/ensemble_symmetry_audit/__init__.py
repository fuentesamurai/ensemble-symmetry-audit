"""Property-based audit of voting ensemble aggregators.

Top-level API:
  - `audit`, `AuditReport`           — hard-voting audit suite
  - `soft_audit`                     — soft-voting audit suite
  - individual detectors             — hard + soft

Optional (requires `pip install ensemble-symmetry-audit[shrink]`):
  - `ensemble_symmetry_audit.strategies`        — Hypothesis strategies
  - `ensemble_symmetry_audit.hypothesis_search` — counterexample shrinking
"""

from .api import audit, AuditReport
from .detectors import (
    DetectorResult,
    balanced_input_symmetry,
    independence_of_irrelevant_alternatives,
    monotonicity,
    null_majority_abstention,
    pareto_unanimity,
    participation_monotonicity,
    permutation_invariance,
    regime_flip_invariance,
    tie_break_determinism,
)
from .soft_api import soft_audit
from .soft_detectors import (
    soft_balanced_input_symmetry,
    soft_continuity,
    soft_monotonicity,
    soft_pareto_unanimity,
    soft_participation_monotonicity,
    soft_permutation_invariance,
    soft_regime_flip_invariance,
)

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
    "participation_monotonicity",
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
    "soft_participation_monotonicity",
    "soft_permutation_invariance",
    "soft_regime_flip_invariance",
]

__version__ = "0.4.0"


def __getattr__(name):
    """Lazy access to optional Hypothesis-backed helpers.

    `from ensemble_symmetry_audit import shrink_hard_counterexample`
    works only if the `[shrink]` extra is installed.
    """
    if name in {
        "shrink_hard_counterexample",
        "shrink_soft_counterexample",
        "strategies",
        "hypothesis_search",
    }:
        if name == "strategies":
            from . import strategies as _strategies
            return _strategies
        if name == "hypothesis_search":
            from . import hypothesis_search as _hs
            return _hs
        from .hypothesis_search import (
            shrink_hard_counterexample,
            shrink_soft_counterexample,
        )
        return locals()[name]
    raise AttributeError(f"module 'ensemble_symmetry_audit' has no attribute {name!r}")
