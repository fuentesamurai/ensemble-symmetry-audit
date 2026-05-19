"""Tests for soft-voting detectors and soft_audit()."""

import numpy as np
import pytest

from ensemble_symmetry_audit import soft_audit
from ensemble_symmetry_audit.soft_detectors import (
    soft_balanced_input_symmetry,
    soft_continuity,
    soft_monotonicity,
    soft_pareto_unanimity,
    soft_permutation_invariance,
    soft_regime_flip_invariance,
)


CLASSES = ["A", "B", "C"]


# --- reference soft aggregators -------------------------------------------

def soft_argmax_average(votes):
    """Average the probability dicts and return the argmax class.

    This is the standard 'soft voting' rule used by scikit-learn's
    VotingClassifier(voting='soft'). Symmetric, monotone, continuous.
    """
    classes = list(votes[0].keys())
    summed = {c: 0.0 for c in classes}
    for v in votes:
        for c, p in v.items():
            summed[c] += p
    return max(summed.items(), key=lambda kv: (kv[1], -classes.index(kv[0])))[0]


def soft_biased_to_A(votes):
    """Adversarial: always returns A regardless of probabilities."""
    return "A"


def soft_first_voter_argmax(votes):
    """Adversarial: returns the argmax of voter 0 only — violates
    permutation invariance and Pareto with the wrong voter."""
    v0 = votes[0]
    return max(v0.items(), key=lambda kv: kv[1])[0]


def soft_anti_monotone(votes):
    """Returns the argmin of the averaged probabilities — violates
    monotonicity."""
    classes = list(votes[0].keys())
    summed = {c: 0.0 for c in classes}
    for v in votes:
        for c, p in v.items():
            summed[c] += p
    return min(summed.items(), key=lambda kv: kv[1])[0]


# --- soft_pareto_unanimity -------------------------------------------------

def test_soft_pareto_passes_for_argmax_average():
    r = soft_pareto_unanimity(soft_argmax_average, CLASSES, n_voters=5,
                              confidence=0.95, seed=0)
    assert r.passed, r


def test_soft_pareto_fails_for_constant():
    r = soft_pareto_unanimity(soft_biased_to_A, CLASSES, n_voters=5,
                              confidence=0.95, seed=0)
    assert not r.passed
    assert r.counterexample is not None
    assert r.counterexample["unanimous_for"] != "A"


# --- soft_balanced_input_symmetry -----------------------------------------

def test_soft_balanced_passes_for_argmax_average():
    r = soft_balanced_input_symmetry(
        soft_argmax_average, CLASSES, n_voters=5, n_trials=2000, seed=0
    )
    assert r.passed, r
    assert r.statistic["test"] == "chi-squared"


def test_soft_balanced_fails_for_constant():
    r = soft_balanced_input_symmetry(
        soft_biased_to_A, CLASSES, n_voters=5, n_trials=500, seed=0
    )
    assert not r.passed


# --- soft_regime_flip_invariance ------------------------------------------

def test_soft_regime_flip_passes_for_argmax_average():
    # Binary case to keep it clean
    binary = ["UP", "DOWN"]
    flip = {"UP": "DOWN", "DOWN": "UP"}
    r = soft_regime_flip_invariance(
        soft_argmax_average, binary, flip, n_voters=5, seed=0
    )
    assert r.passed, r


def test_soft_regime_flip_fails_for_constant():
    binary = ["UP", "DOWN"]
    flip = {"UP": "DOWN", "DOWN": "UP"}
    r = soft_regime_flip_invariance(
        soft_biased_to_A, binary, flip, n_voters=5, seed=0
    )
    # constant aggregator returns "A" for both => flip says expected DOWN
    # but observed A. NB: A is not in binary, so this will fail trivially.
    # Use a binary-compatible constant instead:
    def const_up(votes):
        return "UP"

    r = soft_regime_flip_invariance(
        const_up, binary, flip, n_voters=5, seed=0
    )
    assert not r.passed


# --- soft_monotonicity ---------------------------------------------------

def test_soft_monotonicity_passes_for_argmax_average():
    for target in CLASSES:
        r = soft_monotonicity(
            soft_argmax_average, CLASSES, target, n_voters=5,
            delta=0.05, seed=0,
        )
        assert r.passed, r


def test_soft_monotonicity_fails_for_anti_monotone():
    r = soft_monotonicity(
        soft_anti_monotone, CLASSES, "A", n_voters=5,
        delta=0.10, seed=0,
    )
    assert not r.passed
    assert r.counterexample is not None


# --- soft_permutation_invariance ------------------------------------------

def test_soft_permutation_passes_for_argmax_average():
    r = soft_permutation_invariance(
        soft_argmax_average, CLASSES, n_voters=5, seed=0
    )
    assert r.passed, r


def test_soft_permutation_fails_for_first_voter_only():
    r = soft_permutation_invariance(
        soft_first_voter_argmax, CLASSES, n_voters=5, seed=0
    )
    assert not r.passed
    assert r.counterexample is not None


# --- soft_continuity -----------------------------------------------------

def test_soft_continuity_passes_for_argmax_average():
    r = soft_continuity(
        soft_argmax_average, CLASSES, n_voters=5,
        n_trials=500, epsilon=1e-3, seed=0,
    )
    assert r.passed, r


def test_soft_continuity_runs_for_constant():
    # Constant aggregator is trivially continuous (always returns A)
    r = soft_continuity(
        soft_biased_to_A, CLASSES, n_voters=5,
        n_trials=200, epsilon=1e-3, seed=0,
    )
    assert r.passed


# --- soft_audit() integration --------------------------------------------

def test_soft_audit_runs_all_detectors():
    binary = ["UP", "DOWN"]
    flip = {"UP": "DOWN", "DOWN": "UP"}
    report = soft_audit(
        soft_argmax_average, binary, n_voters=5,
        flip_map=flip, seed=0,
    )
    names = [r.name for r in report.results]
    assert "soft_pareto_unanimity" in names
    assert "soft_balanced_input_symmetry" in names
    assert "soft_regime_flip_invariance" in names
    assert "soft_permutation_invariance" in names
    assert "soft_continuity" in names
    assert any(n.startswith("soft_monotonicity") for n in names)


def test_soft_audit_passes_for_argmax_average():
    binary = ["UP", "DOWN"]
    flip = {"UP": "DOWN", "DOWN": "UP"}
    report = soft_audit(
        soft_argmax_average, binary, n_voters=5,
        flip_map=flip, seed=0,
    )
    assert report.all_passed, str(report)
