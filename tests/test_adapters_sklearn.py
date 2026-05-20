"""Tests for the sklearn adapters."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from ensemble_symmetry_audit import audit, soft_audit, audit_sklearn_classifier
from ensemble_symmetry_audit.adapters.sklearn import (
    adapt,
    adapt_argmax_proba_classifier,
    adapt_bagging_classifier,
    adapt_extra_trees,
    adapt_random_forest,
    adapt_voting_classifier,
)


def make_data(n_classes=3, seed=0):
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X, y


@pytest.fixture
def voting_hard():
    X, y = make_data(n_classes=3)
    clf = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=500, random_state=0)),
            ("dt", DecisionTreeClassifier(random_state=0)),
            ("nb", GaussianNB()),
        ],
        voting="hard",
    )
    clf.fit(X, y)
    return clf


@pytest.fixture
def voting_soft():
    X, y = make_data(n_classes=3)
    clf = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=500, random_state=0)),
            ("dt", DecisionTreeClassifier(random_state=0)),
            ("nb", GaussianNB()),
        ],
        voting="soft",
    )
    clf.fit(X, y)
    return clf


@pytest.fixture
def rf():
    X, y = make_data(n_classes=3)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    return clf


def test_adapt_voting_hard_returns_callable(voting_hard):
    agg = adapt_voting_classifier(voting_hard)
    out = agg([0, 1, 2])
    assert out in {0, 1, 2}


def test_adapt_voting_hard_reproduces_sklearn_tiebreak(voting_hard):
    # With votes [0, 1, 2] all tied, sklearn returns class 0 (lowest index)
    agg = adapt_voting_classifier(voting_hard)
    assert agg([0, 1, 2]) == 0


def test_adapt_voting_soft_returns_argmax(voting_soft):
    agg = adapt_voting_classifier(voting_soft)
    # All voters put 1.0 mass on class 1 -> should return 1
    votes = [{0: 0.05, 1: 0.9, 2: 0.05}] * 3
    assert agg(votes) == 1


def test_adapt_random_forest_is_soft(rf):
    agg = adapt_random_forest(rf)
    votes = [{0: 0.1, 1: 0.8, 2: 0.1}] * 5
    assert agg(votes) == 1


def test_adapt_bagging_chooses_correct_mode():
    X, y = make_data(n_classes=3)
    # GaussianNB has predict_proba -> bagging should use soft path
    clf_soft = BaggingClassifier(
        estimator=GaussianNB(), n_estimators=5, random_state=0
    )
    clf_soft.fit(X, y)
    agg = adapt_bagging_classifier(clf_soft)
    # Soft adapter accepts dicts
    out = agg([{0: 0.1, 1: 0.8, 2: 0.1}] * 5)
    assert out == 1


def test_adapt_argmax_proba_generic_works_for_rf(rf):
    agg = adapt_argmax_proba_classifier(rf)
    votes = [{0: 0.9, 1: 0.05, 2: 0.05}] * 5
    assert agg(votes) == 0


def test_adapt_auto_dispatch(voting_hard, voting_soft, rf):
    # Auto-dispatch should pick correct adapter for each
    agg_h = adapt(voting_hard)
    agg_s = adapt(voting_soft)
    agg_rf = adapt(rf)
    assert agg_h([0, 1, 2]) in {0, 1, 2}
    assert agg_s([{0: 0.1, 1: 0.8, 2: 0.1}] * 3) == 1
    assert agg_rf([{0: 0.1, 1: 0.8, 2: 0.1}] * 5) == 1


def test_adapt_unfitted_raises():
    clf = RandomForestClassifier()
    with pytest.raises(ValueError):
        adapt_random_forest(clf)


def test_audit_sklearn_classifier_voting_hard(voting_hard):
    report = audit_sklearn_classifier(voting_hard, seed=42)
    # Should fail at least balance + IIA on small K, small N
    assert not report.all_passed
    names = [r.name for r in report.results]
    assert "pareto_unanimity" in names
    assert "balanced_input_symmetry" in names


def test_audit_sklearn_classifier_voting_soft(voting_soft):
    report = audit_sklearn_classifier(voting_soft, seed=42)
    # Soft voting passes most properties
    names = [r.name for r in report.results]
    assert "soft_pareto_unanimity" in names
    assert "soft_balanced_input_symmetry" in names


def test_audit_sklearn_classifier_random_forest(rf):
    report = audit_sklearn_classifier(rf, seed=42)
    # Random forest uses soft pipeline -> soft audit
    names = [r.name for r in report.results]
    assert any("soft_" in n for n in names)


def test_audit_sklearn_classifier_no_voters_raises():
    # Classifier without estimators_ attribute and no n_voters arg
    clf = LogisticRegression()
    X, y = make_data(n_classes=3)
    clf.fit(X, y)
    # LogisticRegression has classes_ and predict_proba, so adapt_argmax
    # works, but it has no estimators_, so n_voters must be provided.
    with pytest.raises(ValueError):
        audit_sklearn_classifier(clf, seed=42)


def test_audit_sklearn_classifier_with_explicit_n_voters():
    clf = LogisticRegression()
    X, y = make_data(n_classes=3)
    clf.fit(X, y)
    report = audit_sklearn_classifier(clf, n_voters=5, seed=42)
    # Pipeline ran without error
    assert len(report.results) > 0
