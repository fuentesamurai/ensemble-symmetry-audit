"""Adapters for scikit-learn ensemble classifiers.

Each adapter takes a trained sklearn classifier and returns a callable
that reproduces the classifier's aggregation rule on a list of votes.
The callable is then passed directly to `audit()` or `soft_audit()`.

Hard adapters return functions of type::

    List[label] -> label

Soft adapters return functions of type::

    List[Dict[label, prob]] -> label

The `audit_sklearn_classifier` convenience wraps the adapter, picks the
right audit suite (hard vs soft), and runs it in one call.

sklearn is an optional dependency. Importing this module without
scikit-learn installed raises a friendly ImportError.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ensemble_symmetry_audit.adapters.sklearn requires numpy. "
        "Install with: pip install ensemble-symmetry-audit"
    ) from e

try:
    from sklearn.ensemble import (
        BaggingClassifier,
        ExtraTreesClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ensemble_symmetry_audit.adapters.sklearn requires scikit-learn. "
        "Install with: pip install ensemble-symmetry-audit[sklearn]"
    ) from e


# ---------------------------------------------------------------------------
# Low-level adapters
# ---------------------------------------------------------------------------

def _hard_argmax_bincount(classes: np.ndarray) -> Callable:
    """Reproduce sklearn's hard-voting aggregation:
    `np.argmax(np.bincount(predictions))`.

    Maps each input label to its index in `classes`, calls
    `np.argmax(np.bincount(...))`, returns the label at that index.
    """
    classes = np.asarray(classes)
    class_to_idx = {c: i for i, c in enumerate(classes.tolist())}

    def aggregate(votes):
        try:
            indices = np.array([class_to_idx[v] for v in votes], dtype=int)
        except KeyError as e:
            raise ValueError(
                f"Vote {e.args[0]!r} not in classifier classes {classes.tolist()}"
            )
        idx = int(np.argmax(np.bincount(indices, minlength=len(classes))))
        return classes[idx].item() if hasattr(classes[idx], "item") else classes[idx]

    return aggregate


def _soft_argmax_average(classes: np.ndarray) -> Callable:
    """Reproduce sklearn's soft-voting aggregation:
    average per-class probabilities across voters, then `np.argmax`.
    """
    classes_list = list(classes.tolist() if hasattr(classes, "tolist") else classes)

    def aggregate(votes):
        n = len(classes_list)
        summed = np.zeros(n)
        for v in votes:
            for i, c in enumerate(classes_list):
                summed[i] += v.get(c, 0.0)
        idx = int(np.argmax(summed))
        return classes_list[idx]

    return aggregate


# ---------------------------------------------------------------------------
# High-level per-classifier adapters
# ---------------------------------------------------------------------------

def adapt_voting_classifier(clf: VotingClassifier) -> Callable:
    """Adapter for a trained ``sklearn.ensemble.VotingClassifier``.

    Returns the matching aggregator: hard when ``clf.voting == 'hard'``,
    soft when ``clf.voting == 'soft'``. The aggregator follows the same
    decision rule sklearn applies internally (and inherits its tie-break
    behaviour, which is what makes the audit meaningful).
    """
    if not hasattr(clf, "classes_"):
        raise ValueError(
            "VotingClassifier is not fitted (no `classes_` attribute). "
            "Call clf.fit(X, y) before adapting."
        )
    if clf.voting == "hard":
        return _hard_argmax_bincount(clf.classes_)
    if clf.voting == "soft":
        return _soft_argmax_average(clf.classes_)
    raise ValueError(f"Unknown VotingClassifier voting mode: {clf.voting!r}")


def adapt_bagging_classifier(clf: BaggingClassifier) -> Callable:
    """Adapter for a trained ``sklearn.ensemble.BaggingClassifier``.

    Bagging in sklearn uses hard voting unless every base estimator
    implements ``predict_proba``, in which case ``.predict()`` averages
    probabilities. We pick the right adapter based on whether the
    base estimator exposes ``predict_proba``.
    """
    if not hasattr(clf, "classes_"):
        raise ValueError("BaggingClassifier is not fitted.")
    if hasattr(clf, "estimators_") and clf.estimators_:
        first = clf.estimators_[0]
        soft = hasattr(first, "predict_proba")
    else:
        soft = False
    return (
        _soft_argmax_average(clf.classes_)
        if soft
        else _hard_argmax_bincount(clf.classes_)
    )


def adapt_random_forest(clf: RandomForestClassifier) -> Callable:
    """Adapter for ``sklearn.ensemble.RandomForestClassifier``.

    RandomForest aggregates by averaging per-tree probabilities and
    taking argmax — i.e. soft voting. The returned aggregator
    reproduces that rule.
    """
    if not hasattr(clf, "classes_"):
        raise ValueError("RandomForestClassifier is not fitted.")
    return _soft_argmax_average(clf.classes_)


def adapt_extra_trees(clf: ExtraTreesClassifier) -> Callable:
    """Adapter for ``sklearn.ensemble.ExtraTreesClassifier``.

    Identical aggregation rule to RandomForestClassifier.
    """
    if not hasattr(clf, "classes_"):
        raise ValueError("ExtraTreesClassifier is not fitted.")
    return _soft_argmax_average(clf.classes_)


def adapt_argmax_proba_classifier(clf: Any) -> Callable:
    """Generic adapter for any sklearn-compatible classifier whose
    final decision is ``argmax(predict_proba)``.

    Works with XGBoost, LightGBM, CatBoost, custom estimators, etc.,
    provided they expose a ``classes_`` attribute after fitting. The
    audit then tests the structural properties of the
    argmax-over-averaged-probabilities aggregation step.
    """
    if not hasattr(clf, "classes_"):
        raise ValueError(
            f"{type(clf).__name__} has no `classes_` attribute. "
            "Either the classifier is not fitted, or it does not "
            "expose its class set in the sklearn convention."
        )
    return _soft_argmax_average(clf.classes_)


# ---------------------------------------------------------------------------
# Auto-dispatch + one-call audit convenience
# ---------------------------------------------------------------------------

def adapt(clf: Any) -> Callable:
    """Auto-dispatch adapter.

    Picks the matching specialised adapter for VotingClassifier,
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier;
    falls back to :func:`adapt_argmax_proba_classifier` for anything
    else with a ``predict_proba`` / ``classes_`` interface.
    """
    if isinstance(clf, VotingClassifier):
        return adapt_voting_classifier(clf)
    if isinstance(clf, BaggingClassifier):
        return adapt_bagging_classifier(clf)
    if isinstance(clf, RandomForestClassifier):
        return adapt_random_forest(clf)
    if isinstance(clf, ExtraTreesClassifier):
        return adapt_extra_trees(clf)
    if hasattr(clf, "predict_proba") and hasattr(clf, "classes_"):
        return adapt_argmax_proba_classifier(clf)
    raise TypeError(
        f"No adapter known for {type(clf).__name__}. "
        f"For classifiers exposing predict_proba and classes_, use "
        f"adapt_argmax_proba_classifier(clf) directly."
    )


def audit_sklearn_classifier(
    clf: Any,
    n_voters: int | None = None,
    *,
    flip_map=None,
    neutral_class=None,
    require_abstention: bool = False,
    seed: int = 42,
):
    """One-call audit of a trained sklearn ensemble classifier.

    Picks the right adapter for ``clf`` and runs ``audit()`` (hard
    voting) or ``soft_audit()`` (soft / averaged-probability voting)
    automatically. Returns the resulting ``AuditReport``.

    Parameters
    ----------
    clf
        A trained sklearn classifier (VotingClassifier, Bagging,
        RandomForest, ExtraTrees, or anything else exposing
        ``predict_proba`` and ``classes_``).
    n_voters
        Number of voters per audit input. Defaults to
        ``len(clf.estimators_)`` for VotingClassifier and bagging-like
        classifiers, otherwise required.
    flip_map, neutral_class, require_abstention, seed
        Passed through to ``audit()`` / ``soft_audit()``.
    """
    from ..api import audit
    from ..soft_api import soft_audit

    aggregator = adapt(clf)
    classes = list(clf.classes_.tolist() if hasattr(clf.classes_, "tolist")
                   else clf.classes_)

    if n_voters is None:
        if hasattr(clf, "estimators_") and clf.estimators_:
            n_voters = len(clf.estimators_)
        else:
            raise ValueError(
                "n_voters must be provided when the classifier does "
                "not expose an `estimators_` attribute."
            )

    is_hard = isinstance(clf, VotingClassifier) and clf.voting == "hard"
    if isinstance(clf, BaggingClassifier):
        is_hard = not hasattr(
            clf.estimators_[0] if clf.estimators_ else None,
            "predict_proba",
        )

    if is_hard:
        return audit(
            aggregator,
            classes=classes,
            n_voters=n_voters,
            flip_map=flip_map,
            neutral_class=neutral_class,
            require_abstention=require_abstention,
            seed=seed,
        )
    return soft_audit(
        aggregator,
        classes=classes,
        n_voters=n_voters,
        flip_map=flip_map,
        seed=seed,
    )
