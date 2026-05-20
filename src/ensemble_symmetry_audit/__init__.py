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
    min_n_trials_for_balance,
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
    "min_n_trials_for_balance",
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

__version__ = "0.5.1"


def __getattr__(name):
    """Lazy access to optional sub-packages.

    Hypothesis-backed helpers require ``[shrink]``; sklearn adapters
    require ``[sklearn]``; XGBoost / LightGBM adapters require their
    respective extras.
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
    if name == "adapters":
        from . import adapters as _adapters
        return _adapters
    if name == "audit_sklearn_classifier":
        from .adapters.sklearn import audit_sklearn_classifier
        return audit_sklearn_classifier
    raise AttributeError(f"module 'ensemble_symmetry_audit' has no attribute {name!r}")
