"""Eight property-based detectors for voting-ensemble aggregators.

Each detector takes a `vote_fn` (a callable that maps a sequence of votes
to an aggregated decision) and a small configuration, runs many random
trials, and reports whether a structural property is satisfied. When a
property fails, the detector returns a minimal counterexample so the
caller can reproduce the issue.

Where applicable, detectors report a p-value from a formal statistical
test (chi-squared for balance, binomial for abstention rate) rather
than a hand-picked tolerance, so the audit conclusions are reproducible
and defensible.

The six historical detectors:
  - balanced_input_symmetry
  - regime_flip_invariance
  - null_majority_abstention  (opt-in: only included by `audit()` when
                               an explicit neutral class is given)
  - monotonicity
  - permutation_invariance
  - tie_break_determinism

The two added in v0.2.0 (canonical properties from social-choice theory):
  - pareto_unanimity                       (May 1952, Arrow 1951)
  - independence_of_irrelevant_alternatives (Arrow 1951)
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from scipy.stats import chisquare, binomtest

Vote = Any
VoteFunction = Callable[[Sequence[Vote]], Vote]


@dataclass
class DetectorResult:
    name: str
    passed: bool
    cases_tested: int
    counterexample: dict | None = None
    statistic: dict | None = None
    notes: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        line = f"[{status}] {self.name}  ({self.cases_tested} cases)"
        if self.statistic:
            stat_str = ", ".join(f"{k}={v}" for k, v in self.statistic.items())
            line += f"\n         statistic: {stat_str}"
        if self.counterexample:
            line += f"\n         counterexample: {self.counterexample}"
        return line

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "cases_tested": self.cases_tested,
            "counterexample": self.counterexample,
            "statistic": self.statistic,
            "notes": self.notes,
        }


def _tally(votes: Sequence[Vote]) -> Counter:
    return Counter(votes)


# ---------------------------------------------------------------------------
# 1. Balanced-input symmetry  (chi-squared)
# ---------------------------------------------------------------------------

def balanced_input_symmetry(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    n_voters: int,
    n_trials: int = 2000,
    alpha: float = 0.01,
    seed: int = 42,
) -> DetectorResult:
    """Uniformly random inputs should produce an output distribution
    indistinguishable from uniform.

    Samples `n_trials` vote sets where each voter picks a class uniformly
    at random, then runs a chi-squared goodness-of-fit test against the
    uniform distribution. The detector fails when the test rejects the
    null at level `alpha` (default 1%).
    """
    rng = random.Random(seed)
    outputs: list[Vote] = []
    for _ in range(n_trials):
        votes = [rng.choice(classes) for _ in range(n_voters)]
        outputs.append(vote_fn(votes))

    counts = _tally(outputs)
    observed = [counts.get(c, 0) for c in classes]
    expected = [n_trials / len(classes)] * len(classes)

    res = chisquare(f_obs=observed, f_exp=expected)
    chi2, p = float(res.statistic), float(res.pvalue)
    passed = p >= alpha

    counterexample = None if passed else {
        "observed_counts": dict(zip(classes, observed)),
        "expected_per_class": expected[0],
    }
    return DetectorResult(
        name="balanced_input_symmetry",
        passed=passed,
        cases_tested=n_trials,
        counterexample=counterexample,
        statistic={
            "test": "chi-squared",
            "chi2": round(chi2, 3),
            "p_value": round(p, 4),
            "alpha": alpha,
            "df": len(classes) - 1,
        },
        notes="Uniform input distribution should yield uniform output (chi-squared test).",
    )


# ---------------------------------------------------------------------------
# 2. Regime-flip invariance
# ---------------------------------------------------------------------------

def regime_flip_invariance(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    flip_map: Mapping[Vote, Vote],
    n_voters: int,
    n_trials: int = 500,
    seed: int = 42,
) -> DetectorResult:
    """The aggregator should commute with a symmetric label flip."""
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


# ---------------------------------------------------------------------------
# 3. Null-majority abstention  (binomial test; opt-in)
# ---------------------------------------------------------------------------

def null_majority_abstention(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    neutral_class: Vote,
    n_voters: int,
    n_trials: int = 500,
    expected_rate: float = 0.5,
    alpha: float = 0.01,
    seed: int = 42,
) -> DetectorResult:
    """Balanced non-neutral inputs should resolve to the neutral class
    at a rate not significantly below `expected_rate`.

    This property is *opt-in* (only included by `audit()` when the caller
    passes `neutral_class` explicitly). Many domains require an actionable
    output and cannot abstain, in which case the property does not apply.
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

    res = binomtest(neutrals, n_trials, p=expected_rate, alternative="less")
    p = float(res.pvalue)
    passed = p >= alpha
    rate = neutrals / n_trials

    return DetectorResult(
        name="null_majority_abstention",
        passed=passed,
        cases_tested=n_trials,
        counterexample=None if passed else {
            "neutral_rate": round(rate, 3),
            "expected_rate": expected_rate,
        },
        statistic={
            "test": "binomial",
            "neutral_rate": round(rate, 3),
            "expected_rate": expected_rate,
            "p_value": round(p, 4),
            "alpha": alpha,
            "alternative": "less",
        },
        notes="Opt-in. Balanced non-neutral inputs should resolve to neutral often enough.",
    )


# ---------------------------------------------------------------------------
# 4. Monotonicity
# ---------------------------------------------------------------------------

def monotonicity(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    target_class: Vote,
    n_voters: int,
    n_trials: int = 200,
    seed: int = 42,
) -> DetectorResult:
    """Adding a vote for X must never move the decision away from X."""
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
        notes="Adding a vote for a class must not reduce its chance of winning.",
    )


# ---------------------------------------------------------------------------
# 5. Permutation invariance
# ---------------------------------------------------------------------------

def permutation_invariance(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    n_voters: int,
    n_trials: int = 200,
    n_shuffles: int = 5,
    seed: int = 42,
) -> DetectorResult:
    """Output should depend on the multiset of votes, not voter order."""
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
        notes="Output must depend on the vote multiset, not voter order.",
    )


# ---------------------------------------------------------------------------
# 6. Tie-break determinism
# ---------------------------------------------------------------------------

def tie_break_determinism(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    n_voters: int,
    n_trials: int = 200,
    n_repeats: int = 5,
    seed: int = 42,
) -> DetectorResult:
    """Ties must resolve identically across repeated calls."""
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


# ---------------------------------------------------------------------------
# 7. Pareto unanimity  (NEW in v0.2.0)
# ---------------------------------------------------------------------------

def pareto_unanimity(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    n_voters: int,
    n_trials: int = 200,
    seed: int = 42,
) -> DetectorResult:
    """If every voter votes X, the aggregator must return X.

    This is the Pareto / unanimity condition from social choice theory
    (May 1952; Arrow 1951). It is the most basic sanity property a
    voting rule should satisfy: aggregators that violate it are
    overriding the unanimous will of the voters, usually because of
    miscalibrated weights, threshold bugs, or hidden defaults.
    """
    failures: list[dict] = []
    for c in classes:
        for _ in range(max(n_trials // max(len(classes), 1), 1)):
            votes = [c] * n_voters
            out = vote_fn(votes)
            if out != c:
                failures.append({
                    "unanimous_for": c,
                    "votes": votes,
                    "output": out,
                })
                break
        if len(failures) >= 3:
            break

    cases = n_trials
    passed = not failures
    return DetectorResult(
        name="pareto_unanimity",
        passed=passed,
        cases_tested=cases,
        counterexample=failures[0] if failures else None,
        notes="Unanimous input must yield the unanimous choice (Pareto / May 1952).",
    )


# ---------------------------------------------------------------------------
# 8. Independence of Irrelevant Alternatives  (NEW in v0.2.0)
# ---------------------------------------------------------------------------

def independence_of_irrelevant_alternatives(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    n_voters: int,
    n_trials: int = 200,
    seed: int = 42,
) -> DetectorResult:
    """Removing a losing class should not change the choice between the
    remaining classes.

    Arrow's IIA in its applied form: take any random ballot v, find its
    winner w = vote_fn(v). Pick any class L that is strictly behind w in
    the vote tally (a "loser"). Replace every L vote in v with a uniformly
    chosen non-L class to obtain v'. The winner of v' should still be w.

    Arrow's theorem (1951) proves that no deterministic aggregator over
    three or more classes can simultaneously satisfy unanimity, IIA, and
    non-dictatorship. This detector therefore *expects* most non-trivial
    3+ class aggregators to fail it. The point of running it is to
    quantify and locate the violations, not to be surprised by their
    existence.
    """
    if len(classes) < 3:
        return DetectorResult(
            name="independence_of_irrelevant_alternatives",
            passed=True,
            cases_tested=0,
            notes="IIA is trivially satisfied with fewer than 3 classes — skipped.",
        )

    rng = random.Random(seed)
    failures: list[dict] = []
    for _ in range(n_trials):
        votes = [rng.choice(classes) for _ in range(n_voters)]
        winner = vote_fn(votes)
        counts = _tally(votes)
        winner_count = counts.get(winner, 0)
        losers = [c for c in classes if c != winner and counts.get(c, 0) < winner_count]
        if not losers:
            continue
        loser = rng.choice(losers)
        replacements = [c for c in classes if c != loser]
        new_votes = [
            (rng.choice(replacements) if v == loser else v) for v in votes
        ]
        new_winner = vote_fn(new_votes)
        if new_winner != winner:
            failures.append({
                "original_votes": votes,
                "winner": winner,
                "removed_loser": loser,
                "new_votes": new_votes,
                "new_winner": new_winner,
            })
            if len(failures) >= 3:
                break

    passed = not failures
    return DetectorResult(
        name="independence_of_irrelevant_alternatives",
        passed=passed,
        cases_tested=n_trials,
        counterexample=failures[0] if failures else None,
        notes=(
            "Replacing losing-class votes with other non-winning classes should "
            "not change the winner (Arrow's IIA, 1951). Expect this to fail for "
            "most non-trivial 3+ class aggregators — Arrow's theorem guarantees "
            "no deterministic non-dictatorial rule can satisfy IIA, unanimity, "
            "and universal domain simultaneously."
        ),
    )
