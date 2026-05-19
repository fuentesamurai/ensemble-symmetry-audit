"""High-level audit API.

The `audit` function runs every applicable detector against a voting
function and bundles the results into a single `AuditReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .detectors import (
    DetectorResult,
    balanced_input_symmetry,
    monotonicity,
    null_majority_abstention,
    permutation_invariance,
    regime_flip_invariance,
    tie_break_determinism,
)


@dataclass
class AuditReport:
    results: list[DetectorResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> list[DetectorResult]:
        return [r for r in self.results if not r.passed]

    def __str__(self) -> str:
        lines = ["Ensemble bias audit report", "-" * 40]
        lines.extend(str(r) for r in self.results)
        lines.append("-" * 40)
        lines.append("ALL PROPERTIES HELD" if self.all_passed
                     else f"{len(self.failed)} property failure(s) detected")
        return "\n".join(lines)


def audit(
    vote_fn: Callable[[Sequence[Any]], Any],
    classes: Sequence[Any],
    n_voters: int,
    *,
    neutral_class: Any = None,
    flip_map: Mapping[Any, Any] | None = None,
    seed: int = 42,
) -> AuditReport:
    """Run the full battery of bias detectors against `vote_fn`.

    Parameters
    ----------
    vote_fn
        A callable that takes a list of votes and returns an aggregated
        decision.
    classes
        The full set of possible vote / decision labels.
    n_voters
        Number of voters in each test input.
    neutral_class
        If provided, enables `null_majority_abstention` and excludes
        the neutral class from monotonicity tests.
    flip_map
        Optional dict describing a symmetric label-flip
        (e.g. {"BUY": "SELL", "SELL": "BUY", "HOLD": "HOLD"}).
        Enables `regime_flip_invariance`.
    seed
        Base seed for reproducibility. Each detector derives its own
        seed from this value.
    """
    report = AuditReport()
    report.results.append(
        balanced_input_symmetry(vote_fn, classes, n_voters, seed=seed)
    )
    if flip_map is not None:
        report.results.append(
            regime_flip_invariance(
                vote_fn, classes, flip_map, n_voters, seed=seed + 1
            )
        )
    if neutral_class is not None:
        report.results.append(
            null_majority_abstention(
                vote_fn, classes, neutral_class, n_voters, seed=seed + 2
            )
        )
    for target in classes:
        if target == neutral_class:
            continue
        report.results.append(
            monotonicity(vote_fn, classes, target, n_voters, seed=seed + 3)
        )
    report.results.append(
        permutation_invariance(vote_fn, classes, n_voters, seed=seed + 4)
    )
    report.results.append(
        tie_break_determinism(vote_fn, classes, n_voters, seed=seed + 5)
    )
    return report
