"""High-level audit API.

The `audit` function runs every applicable detector against a voting
function and bundles the results into a single `AuditReport`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .detectors import (
    DetectorResult,
    balanced_input_symmetry,
    independence_of_irrelevant_alternatives,
    monotonicity,
    null_majority_abstention,
    pareto_unanimity,
    participation_monotonicity,
    permutation_invariance,
    regime_flip_invariance,
    tie_break_determinism,
)


@dataclass
class AuditReport:
    results: list[DetectorResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> list[DetectorResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "all_passed": self.all_passed,
            "n_failed": len(self.failed),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        lines = ["Ensemble symmetry audit report", "-" * 50]
        lines.extend(str(r) for r in self.results)
        lines.append("-" * 50)
        lines.append(
            "ALL PROPERTIES HELD" if self.all_passed
            else f"{len(self.failed)} property failure(s) detected"
        )
        return "\n".join(lines)


def audit(
    vote_fn: Callable[[Sequence[Any]], Any],
    classes: Sequence[Any],
    n_voters: int,
    *,
    neutral_class: Any = None,
    flip_map: Mapping[Any, Any] | None = None,
    require_abstention: bool = False,
    seed: int = 42,
) -> AuditReport:
    """Run the full battery of property detectors against `vote_fn`.

    Parameters
    ----------
    vote_fn
        Callable mapping a list of votes to an aggregated decision.
    classes
        Full set of possible vote / decision labels.
    n_voters
        Number of voters per test input.
    neutral_class
        Optional neutral class. Used by monotonicity to skip the
        neutral target and (only when `require_abstention=True`) by
        `null_majority_abstention`.
    flip_map
        Optional label-flip mapping (e.g. {"BUY": "SELL", "SELL": "BUY",
        "HOLD": "HOLD"}). Enables `regime_flip_invariance`.
    require_abstention
        If True, includes `null_majority_abstention`. Many domains
        require an actionable decision and cannot abstain, so the
        property is opt-in. Requires `neutral_class`.
    seed
        Base seed for reproducibility. Each detector derives its own
        seed from this value.
    """
    report = AuditReport(config={
        "classes": list(classes),
        "n_voters": n_voters,
        "neutral_class": neutral_class,
        "flip_map": dict(flip_map) if flip_map else None,
        "require_abstention": require_abstention,
        "seed": seed,
    })

    report.results.append(
        pareto_unanimity(vote_fn, classes, n_voters, seed=seed)
    )
    report.results.append(
        balanced_input_symmetry(vote_fn, classes, n_voters, seed=seed + 1)
    )
    if flip_map is not None:
        report.results.append(
            regime_flip_invariance(
                vote_fn, classes, flip_map, n_voters, seed=seed + 2
            )
        )
    if require_abstention:
        if neutral_class is None:
            raise ValueError(
                "require_abstention=True requires neutral_class to be set."
            )
        report.results.append(
            null_majority_abstention(
                vote_fn, classes, neutral_class, n_voters, seed=seed + 3
            )
        )
    for target in classes:
        if target == neutral_class:
            continue
        report.results.append(
            monotonicity(vote_fn, classes, target, n_voters, seed=seed + 4)
        )
        report.results.append(
            participation_monotonicity(
                vote_fn, classes, target, n_voters, seed=seed + 8,
            )
        )
    report.results.append(
        permutation_invariance(vote_fn, classes, n_voters, seed=seed + 5)
    )
    report.results.append(
        tie_break_determinism(vote_fn, classes, n_voters, seed=seed + 6)
    )
    report.results.append(
        independence_of_irrelevant_alternatives(
            vote_fn, classes, n_voters, seed=seed + 7
        )
    )
    return report
