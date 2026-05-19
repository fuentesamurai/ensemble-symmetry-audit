"""Tests for the Hypothesis-based counterexample shrinker."""

from collections import Counter

from ensemble_symmetry_audit.hypothesis_search import (
    shrink_hard_counterexample,
    shrink_soft_counterexample,
)


def biased_majority(votes):
    """Always returns 'A' when input contains any 'A' vote."""
    if "A" in votes:
        return "A"
    return votes[0]


def test_shrink_hard_finds_minimal_violation():
    classes = ["A", "B", "C"]

    def violates_pareto_B(votes):
        # Pareto for B: all-B input should return B
        return all(v == "B" for v in votes) and biased_majority(votes) != "B"

    # Wait — biased_majority returns first vote when no A. all-B → returns B. OK.
    # So this WON'T violate. Use a real violation:
    def violates_pareto_A_under_const_B(votes):
        return all(v == "A" for v in votes) and "B" != "A"

    # That's always True for all-A input. Hypothesis should find a tiny all-A.
    result = shrink_hard_counterexample(
        violates_pareto_A_under_const_B,
        classes=classes,
        n_voters=3,
        max_examples=100,
    )
    assert result is not None
    assert all(v == "A" for v in result)


def test_shrink_hard_returns_none_when_no_violation():
    classes = ["A", "B"]

    def never_violates(votes):
        return False

    result = shrink_hard_counterexample(
        never_violates,
        classes=classes,
        n_voters=3,
        max_examples=50,
    )
    assert result is None


def test_shrink_soft_finds_violation():
    classes = ["A", "B"]

    def soft_violation(votes):
        # Trivial violation: any vote where A has > 0.5 mass
        return any(v.get("A", 0) > 0.5 for v in votes)

    result = shrink_soft_counterexample(
        soft_violation,
        classes=classes,
        n_voters=2,
        max_examples=50,
    )
    assert result is not None
    assert any(v.get("A", 0) > 0.5 for v in result)
