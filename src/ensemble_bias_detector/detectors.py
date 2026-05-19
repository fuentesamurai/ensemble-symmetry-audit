"""Six property-based detectors for voting-ensemble bias.

Each detector takes a `vote_fn` (a callable that maps a sequence of votes
to an aggregated decision) and a small configuration, runs many random
trials, and reports whether a structural property is satisfied. When a
property fails, the detector returns a minimal counterexample so the
caller can reproduce the issue.

The detectors are pure-Python and dependency-free at runtime. They are
deliberately conservative: they describe behaviours that nearly every
sensible voting aggregator should satisfy, so a failure is almost
always a real signal.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

Vote = Any
VoteFunction = Callable[[Sequence[Vote]], Vote]


@dataclass
class DetectorResult:
    name: str
    passed: bool
    cases_tested: int
    counterexample: dict | None = None
    notes: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        line = f"[{status}] {self.name}  ({self.cases_tested} cases)"
        if self.counterexample:
            line += f"\n         counterexample: {self.counterexample}"
        return line


def _tally(votes: Sequence[Vote]) -> Counter:
    return Counter(votes)


def balanced_input_symmetry(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    n_voters: int,
    n_trials: int = 2000,
    tolerance: float = 0.20,
    seed: int | None = None,
) -> DetectorResult:
    """Uniformly random inputs should produce a roughly uniform output.

    Samples `n_trials` vote sets where each voter picks a class uniformly
    at random, then tallies the aggregator's decisions. If one class
    appears far more often than expected, the aggregator carries a
    structural bias toward that class.
    """
    rng = random.Random(seed)
    outputs: list[Vote] = []
    for _ in range(n_trials):
        votes = [rng.choice(classes) for _ in range(n_voters)]
        outputs.append(vote_fn(votes))

    counts = _tally(outputs)
    expected = n_trials / len(classes)
    max_dev = max(abs(counts.get(c, 0) - expected) / expected for c in classes)

    passed = max_dev <= tolerance
    counterexample = None if passed else {
        "observed_counts": dict(counts),
        "expected_per_class": expected,
        "max_relative_deviation": round(max_dev, 3),
        "tolerance": tolerance,
    }
    return DetectorResult(
        name="balanced_input_symmetry",
        passed=passed,
        cases_tested=n_trials,
        counterexample=counterexample,
        notes="Uniform random inputs should yield a uniform output distribution.",
    )


def regime_flip_invariance(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    flip_map: Mapping[Vote, Vote],
    n_voters: int,
    n_trials: int = 500,
    seed: int | None = None,
) -> DetectorResult:
    """The aggregator should commute with a symmetric label flip.

    For every random input v, also evaluate flip(v). If the aggregator
    is unbiased, flip(vote_fn(v)) == vote_fn(flip(v)). A failure means
    the aggregator reacts differently to mirror-image inputs, which is
    a hallmark of an internal asymmetry.
    """
    rng = random.Random(seed)
    failures: list[dict] = []
    for _ in range(n_trials):
        votes = [rng.choice(classes) for _ in range(n_voters)]
        flipped = [flip_map[v] for v in votes]
        out_a = vote_fn(votes)
        out_b = vote_fn(flipped)
        if flip_map.get(out_a) != out_b:
            failures.append({
                "votes": votes,
                "flipped_votes": flipped,
                "out_original": out_a,
                "out_flipped": out_b,
                "expected_flipped": flip_map.get(out_a),
            })
            if len(failures) >= 3:
                break

    passed = not failures
    return DetectorResult(
        name="regime_flip_invariance",
        passed=passed,
        cases_tested=n_trials,
        counterexample=failures[0] if failures else None,
        notes="Symmetric label flips should be equivariant under aggregation.",
    )


def null_majority_abstention(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    neutral_class: Vote,
    n_voters: int,
    n_trials: int = 200,
    tolerance: float = 0.5,
    seed: int | None = None,
) -> DetectorResult:
    """Balanced non-neutral inputs should resolve to the neutral class.

    Constructs inputs that are perfectly balanced across non-neutral
    classes and checks that the aggregator returns the neutral option
    at least `tolerance` of the time. An aggregator that picks sides
    when the evidence is even is making decisions noise cannot justify.
    """
    rng = random.Random(seed)
    non_neutral = [c for c in classes if c != neutral_class]
    if not non_neutral:
        return DetectorResult(
            name="null_majority_abstention",
            passed=True,
            cases_tested=0,
            notes="No non-neutral classes — skipped.",
        )

    neutrals = 0
    for _ in range(n_trials):
        per_class = n_voters // len(non_neutral)
        remainder = n_voters - per_class * len(non_neutral)
        votes: list[Vote] = []
        for c in non_neutral:
            votes.extend([c] * per_class)
        votes.extend(rng.choice(non_neutral) for _ in range(remainder))
        rng.shuffle(votes)
        if vote_fn(votes) == neutral_class:
            neutrals += 1

    rate = neutrals / n_trials
    passed = rate >= tolerance
    return DetectorResult(
        name="null_majority_abstention",
        passed=passed,
        cases_tested=n_trials,
        counterexample=None if passed else {
            "neutral_rate": round(rate, 3),
            "required_rate": tolerance,
        },
        notes="Balanced non-neutral inputs should resolve to the neutral class.",
    )


def monotonicity(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    target_class: Vote,
    n_voters: int,
    n_trials: int = 200,
    seed: int | None = None,
) -> DetectorResult:
    """Adding a vote for X must never move the decision away from X.

    Pairs each random input with the same input plus one extra vote for
    `target_class`. If the aggregator was already returning `target_class`
    and adding more support causes it to switch away, monotonicity is
    violated — a common bug in weighted or threshold-based aggregators.
    """
    rng = random.Random(seed)
    violations: list[dict] = []
    for _ in range(n_trials):
        votes = [rng.choice(classes) for _ in range(max(n_voters - 1, 0))]
        plus = votes + [target_class]
        out_a = vote_fn(votes) if votes else None
        out_b = vote_fn(plus)
        if out_a == target_class and out_b != target_class:
            violations.append({
                "votes": votes,
                "plus_target": plus,
                "out_without_target": out_a,
                "out_with_target": out_b,
            })
            if len(violations) >= 3:
                break

    passed = not violations
    return DetectorResult(
        name=f"monotonicity[{target_class}]",
        passed=passed,
        cases_tested=n_trials,
        counterexample=violations[0] if violations else None,
        notes="Adding a vote for a class must not reduce that class's chance of winning.",
    )


def permutation_invariance(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    n_voters: int,
    n_trials: int = 200,
    n_shuffles: int = 5,
    seed: int | None = None,
) -> DetectorResult:
    """Output should depend on the multiset of votes, not voter order.

    For each random input, the same set of votes is shuffled several
    times and re-evaluated. If the output changes, the aggregator is
    silently treating voters as positional rather than symmetric.
    """
    rng = random.Random(seed)
    failures: list[dict] = []
    for _ in range(n_trials):
        votes = [rng.choice(classes) for _ in range(n_voters)]
        base = vote_fn(list(votes))
        for _ in range(n_shuffles):
            shuf = list(votes)
            rng.shuffle(shuf)
            shuf_out = vote_fn(shuf)
            if shuf_out != base:
                failures.append({
                    "votes": votes,
                    "shuffled": shuf,
                    "out_base": base,
                    "out_shuffled": shuf_out,
                })
                break
        if len(failures) >= 3:
            break

    passed = not failures
    return DetectorResult(
        name="permutation_invariance",
        passed=passed,
        cases_tested=n_trials,
        counterexample=failures[0] if failures else None,
        notes="Output must depend on the vote multiset, not the voter order.",
    )


def tie_break_determinism(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    n_voters: int,
    n_trials: int = 200,
    n_repeats: int = 5,
    seed: int | None = None,
) -> DetectorResult:
    """Ties must resolve identically across repeated calls.

    Only inspects inputs that produce ties at the top of the vote
    tally. Calls the aggregator several times with the same input and
    asserts the result is the same every time. Non-determinism in
    tie-break logic is a common silent bug.
    """
    rng = random.Random(seed)
    failures: list[dict] = []
    for _ in range(n_trials):
        votes = [rng.choice(classes) for _ in range(n_voters)]
        counts = _tally(votes)
        top = max(counts.values())
        tied = [c for c, k in counts.items() if k == top]
        if len(tied) < 2:
            continue
        outputs = {vote_fn(list(votes)) for _ in range(n_repeats)}
        if len(outputs) > 1:
            failures.append({
                "votes": votes,
                "tied_classes": tied,
                "outputs_observed": list(outputs),
            })
            if len(failures) >= 3:
                break

    passed = not failures
    return DetectorResult(
        name="tie_break_determinism",
        passed=passed,
        cases_tested=n_trials,
        counterexample=failures[0] if failures else None,
        notes="Ties must resolve deterministically across identical inputs.",
    )
