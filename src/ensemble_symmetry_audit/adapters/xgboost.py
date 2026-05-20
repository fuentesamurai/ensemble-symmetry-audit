"""Adapter for XGBoost classifiers.

XGBoost is a gradient-boosting ensemble: each booster corrects the
residual of the previous one, so the "voters" are not strictly
independent. From the structural-audit point of view, what we test is
the final aggregation step — ``argmax`` over averaged class
probabilities from each booster — which is the same shape as soft
voting in sklearn.

The adapter is a thin wrapper around
``adapt_argmax_proba_classifier``; XGBoost exposes ``classes_`` after
fitting like any sklearn-compatible classifier.
"""

from __future__ import annotations

from typing import Any, Callable

from .sklearn import adapt_argmax_proba_classifier


def adapt_xgboost_classifier(clf: Any) -> Callable:
    """Adapter for a trained ``xgboost.XGBClassifier``.

    Returns a soft-voting aggregator that reproduces XGBoost's
    final ``argmax(predict_proba)`` decision rule. Suitable for
    :func:`ensemble_symmetry_audit.soft_audit`.

    Raises
    ------
    ImportError
        If xgboost is not installed.
    ValueError
        If the classifier is not fitted (no ``classes_``).
    """
    try:
        import xgboost  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "adapt_xgboost_classifier requires xgboost. "
            "Install with: pip install ensemble-symmetry-audit[xgboost]"
        ) from e

    return adapt_argmax_proba_classifier(clf)
