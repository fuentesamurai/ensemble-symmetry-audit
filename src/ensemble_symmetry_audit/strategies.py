"""Hypothesis strategies for voting inputs.

Two families:
  - hard-vote strategies: lists of categorical class labels
  - soft-vote strategies: lists of probability dictionaries over classes

These are the natural building blocks for property tests that drive the
auditor with Hypothesis. They can also be used independently in user
tests via @given.

Hypothesis is an *optional* runtime dependency. If it is not installed,
importing this module raises a friendly ImportError that points at the
`shrink` extra.
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    from hypothesis import strategies as st
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ensemble_symmetry_audit.strategies requires the optional "
        "`hypothesis` dependency. Install it with:\n\n"
        "    pip install ensemble-symmetry-audit[shrink]\n"
    ) from e


def vote_lists(classes: Sequence[Any], n_voters: int) -> st.SearchStrategy:
    """Strategy: lists of `n_voters` categorical labels drawn from `classes`."""
    return st.lists(
        st.sampled_from(list(classes)),
        min_size=n_voters,
        max_size=n_voters,
    )


def balanced_vote_lists(
    classes: Sequence[Any], n_voters: int
) -> st.SearchStrategy:
    """Strategy: vote lists where every class appears at least once.

    Useful for tests that require non-degenerate inputs (every class
    must be represented).
    """
    if n_voters < len(classes):
        return vote_lists(classes, n_voters)
    base = list(classes)
    extra = st.lists(
        st.sampled_from(list(classes)),
        min_size=n_voters - len(classes),
        max_size=n_voters - len(classes),
    )
    return extra.map(lambda xs: base + xs)


def probability_distributions(classes: Sequence[Any]) -> st.SearchStrategy:
    """Strategy: probability dicts over classes summing to 1.

    Each value is in (0, 1). The smallest representable probability for
    any class is bounded to avoid floating-point degeneracy.
    """
    weights = st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
        min_size=len(classes),
        max_size=len(classes),
    )

    def normalise(w):
        total = sum(w)
        return {c: x / total for c, x in zip(classes, w)}

    return weights.map(normalise)


def probability_vote_lists(
    classes: Sequence[Any], n_voters: int
) -> st.SearchStrategy:
    """Strategy: lists of `n_voters` probability dicts over `classes`."""
    return st.lists(
        probability_distributions(classes),
        min_size=n_voters,
        max_size=n_voters,
    )


def concentrated_probability_vote_lists(
    classes: Sequence[Any],
    n_voters: int,
    target_class: Any,
    confidence: float = 0.9,
) -> st.SearchStrategy:
    """Strategy: probability vote lists where every voter puts
    `confidence` mass on `target_class` and the remainder spread over
    the others. Used by the soft Pareto test.
    """
    if target_class not in classes:
        raise ValueError(f"target_class {target_class!r} not in classes")

    others = [c for c in classes if c != target_class]
    remainder = 1.0 - confidence
    per_other = remainder / max(len(others), 1)

    def make_one():
        d = {c: per_other for c in others}
        d[target_class] = confidence
        return d

    return st.builds(
        lambda _: [make_one() for _ in range(n_voters)],
        st.integers(min_value=0, max_value=0),
    )
