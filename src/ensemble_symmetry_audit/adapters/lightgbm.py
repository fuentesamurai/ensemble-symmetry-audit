"""Adapter for LightGBM classifiers.

Like XGBoost, LightGBM is a gradient-boosting ensemble. The structural
audit applies to the final aggregation step — ``argmax`` over averaged
class probabilities. The adapter is a thin wrapper around
``adapt_argmax_proba_classifier``.
"""

from __future__ import annotations

from typing import Any, Callable

from .sklearn import adapt_argmax_proba_classifier


def adapt_lightgbm_classifier(clf: Any) -> Callable:
    """Adapter for a trained ``lightgbm.LGBMClassifier``.

    Returns a soft-voting aggregator that reproduces LightGBM's final
    ``argmax(predict_proba)`` decision rule. Suitable for
    :func:`ensemble_symmetry_audit.soft_audit`.

    Raises
    ------
    ImportError
        If lightgbm is not installed.
    ValueError
        If the classifier is not fitted (no ``classes_``).
    """
    try:
        import lightgbm  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "adapt_lightgbm_classifier requires lightgbm. "
            "Install with: pip install ensemble-symmetry-audit[lightgbm]"
        ) from e

    return adapt_argmax_proba_classifier(clf)
