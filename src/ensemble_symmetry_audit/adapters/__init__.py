"""First-class adapters for popular ensemble libraries.

Each adapter takes a trained classifier and returns a vote-aggregator
callable that you can pass directly to :func:`audit` or
:func:`soft_audit`.

Sub-modules:
    * ``sklearn``  — VotingClassifier, BaggingClassifier,
      RandomForestClassifier, ExtraTreesClassifier, plus a generic
      ``adapt_argmax_proba_classifier`` and a one-call
      ``audit_sklearn_classifier``.
    * ``xgboost`` — ``adapt_xgboost_classifier``.
    * ``lightgbm`` — ``adapt_lightgbm_classifier``.

sklearn and the boosting libraries are optional dependencies; install
the relevant extras:

    pip install ensemble-symmetry-audit[sklearn]
    pip install ensemble-symmetry-audit[xgboost]
    pip install ensemble-symmetry-audit[lightgbm]

or all of them:

    pip install ensemble-symmetry-audit[adapters]
"""

from __future__ import annotations


def __getattr__(name):
    """Lazy-load adapter submodules so the core package does not
    require sklearn / xgboost / lightgbm at import time."""
    if name == "sklearn":
        from . import sklearn as _sk
        return _sk
    if name == "xgboost":
        from . import xgboost as _xgb
        return _xgb
    if name == "lightgbm":
        from . import lightgbm as _lgb
        return _lgb
    raise AttributeError(
        f"module 'ensemble_symmetry_audit.adapters' has no attribute {name!r}"
    )
