"""High-level audit API for soft-voting aggregators."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from .api import AuditReport
from .soft_detectors import (
    soft_balanced_input_symmetry,
    soft_continuity,
    soft_monotonicity,
    soft_pareto_unanimity,
    soft_participation_monotonicity,
    soft_permutation_invariance,
    soft_regime_flip_invariance,
)

ProbVote = Dict[Any, float]
SoftVoteFunction = Callable[[Sequence[ProbVote]], Any]


def soft_audit(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    n_voters: int,
    *,
    flip_map: Mapping[Any, Any] | None = None,
    pareto_confidence: float = 0.95,
    continuity_epsilon: float = 1e-3,
    seed: int = 42,
) -> AuditReport:
    """Run the soft-voting audit suite against `vote_fn`.

    Parameters
    ----------
    vote_fn
        Callable mapping a list of probability dicts (one per voter) to
        a single chosen class label.
    classes
        Full set of classes.
    n_voters
        Number of voters per test input.
    flip_map
        Optional label-flip mapping. Enables `soft_regime_flip_invariance`.
    pareto_confidence
        Probability mass threshold for the soft Pareto test (default 0.95).
    continuity_epsilon
        Magnitude of probability perturbation for the continuity test
        (default 1e-3).
    seed
        Base seed for reproducibility.
    """
    report = AuditReport(config={
        "kind": "soft",
        "classes": list(classes),
        "n_voters": n_voters,
        "flip_map": dict(flip_map) if flip_map else None,
        "pareto_confidence": pareto_confidence,
        "continuity_epsilon": continuity_epsilon,
        "seed": seed,
    })

    report.results.append(
        soft_pareto_unanimity(
            vote_fn, classes, n_voters,
            confidence=pareto_confidence, seed=seed,
        )
    )
    report.results.append(
        soft_balanced_input_symmetry(
            vote_fn, classes, n_voters, seed=seed + 1
        )
    )
    if flip_map is not None:
        report.results.append(
            soft_regime_flip_invariance(
                vote_fn, classes, flip_map, n_voters, seed=seed + 2
            )
        )
    for target in classes:
        report.results.append(
            soft_monotonicity(
                vote_fn, classes, target, n_voters, seed=seed + 3
            )
        )
        report.results.append(
            soft_participation_monotonicity(
                vote_fn, classes, target, n_voters, seed=seed + 6
            )
        )
    report.results.append(
        soft_permutation_invariance(
            vote_fn, classes, n_voters, seed=seed + 4
        )
    )
    report.results.append(
        soft_continuity(
            vote_fn, classes, n_voters,
            epsilon=continuity_epsilon, seed=seed + 5,
        )
    )
    return report
