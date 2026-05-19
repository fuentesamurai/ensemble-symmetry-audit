"""Hypothesis-driven adversarial counterexample search.

Wraps `hypothesis.find()` so a caller can search for the smallest input
that violates a user-supplied property, getting Hypothesis's directed
search + automatic shrinking for free.

This is intentionally a thin layer — the rest of the library relies on
random sampling, which is fast and sufficient for routine audits. When
you have a property failure and want a *minimal* counterexample, this
module is the tool to call.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from hypothesis import HealthCheck, find, settings
from hypothesis import strategies as st

from .strategies import (
    probability_vote_lists,
    vote_lists,
)


def shrink_hard_counterexample(
    is_violation: Callable[[list], bool],
    classes: Sequence[Any],
    n_voters: int,
    max_examples: int = 200,
    deadline_ms: int | None = 5000,
) -> list | None:
    """Search for the smallest hard-vote list that satisfies `is_violation`.

    `is_violation(votes) -> bool` should return True when the given vote
    list breaks the property under audit. Returns the minimal violating
    example, or None if no violation is found within `max_examples`
    attempts.
    """
    s = vote_lists(classes, n_voters)
    return _safe_find(is_violation, s, max_examples, deadline_ms)


def shrink_soft_counterexample(
    is_violation: Callable[[list], bool],
    classes: Sequence[Any],
    n_voters: int,
    max_examples: int = 200,
    deadline_ms: int | None = 10000,
) -> list | None:
    """Search for the smallest soft-vote list that satisfies `is_violation`.

    `is_violation(votes) -> bool` should return True when the given list
    of probability dicts breaks the property. Returns the minimal
    violating example, or None.
    """
    s = probability_vote_lists(classes, n_voters)
    return _safe_find(is_violation, s, max_examples, deadline_ms)


def _safe_find(
    is_violation: Callable[[Any], bool],
    strategy: st.SearchStrategy,
    max_examples: int,
    deadline_ms: int | None,
) -> Any | None:
    s = settings(
        max_examples=max_examples,
        deadline=deadline_ms,
        suppress_health_check=[HealthCheck.too_slow,
                               HealthCheck.data_too_large],
    )
    try:
        return find(strategy, is_violation, settings=s)
    except Exception:
        # NoSuchExample, deadline, etc.
        return None
