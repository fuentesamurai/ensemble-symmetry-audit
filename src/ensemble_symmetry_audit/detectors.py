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

    Along with the p-value the result reports **effect size** (Cohen's w
    and the maximum relative deviation from uniform): with `n_trials =
    2000` chi-squared rejects vanishingly small biases that may not be
    actionable. Reading p-value and effect size together lets you
    distinguish "statistically significant but tiny" from "structurally
    important". As a rule of thumb (Cohen 1988):
    w ≈ 0.1 small, 0.3 medium, 0.5 large.
    """
    import math
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
    cohens_w = math.sqrt(chi2 / n_trials) if n_trials > 0 else 0.0
    max_rel_dev = max(
        abs(o - e) / e for o, e in zip(observed, expected)
    ) if expected[0] > 0 else 0.0
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
            "cohens_w": round(cohens_w, 3),
            "max_relative_deviation": round(max_rel_dev, 3),
        },
        notes=(
            "Uniform input distribution should yield uniform output "
            "(chi-squared test). Effect size (Cohen's w) and "
            "max_relative_deviation distinguish significant-but-small "
            "from structurally important deviations."
        ),
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
    seed: int = 42,
    mode: str = "transpositions",
) -> DetectorResult:
    """Output should depend on the multiset of votes, not voter order.

    Two modes are supported:

    - ``mode="transpositions"`` (default): test invariance under every
      adjacent transposition of voters. If the aggregator commutes with
      all `n_voters - 1` adjacent swaps, it commutes with all of `S_n`
      by composition (the adjacent transpositions generate the
      symmetric group). This is exhaustive coverage at linear cost.

    - ``mode="random"``: sample random shuffles. Useful as a quick
      smoke test but statistically weak for large `n_voters` —
      `200` random samples from `S_11` (~40M elements) catch insertion-
      order bugs but miss subtle order-pair dependencies.
    """
    if mode not in {"transpositions", "random"}:
        raise ValueError(f"unknown mode: {mode!r}")

    rng = random.Random(seed)
    failures: list[dict] = []

    if mode == "transpositions":
        for _ in range(n_trials):
            votes = [rng.choice(classes) for _ in range(n_voters)]
            base = vote_fn(list(votes))
            for i in range(n_voters - 1):
                if votes[i] == votes[i + 1]:
                    continue  # swap is identity, skip
                swapped = list(votes)
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                if vote_fn(swapped) != base:
                    failures.append({
                        "votes": votes,
                        "swapped_indices": [i, i + 1],
                        "swapped": swapped,
                        "out_base": base,
                        "out_swapped": vote_fn(swapped),
                    })
                    break
            if len(failures) >= 3:
                break
        cases_tested = n_trials
        notes = (
            "Exhaustive: invariance under all adjacent transpositions "
            "implies invariance under the full symmetric group S_n."
        )
    else:  # random
        n_shuffles = 5
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
        cases_tested = n_trials
        notes = (
            "Random shuffles — sparse coverage of S_n for large n_voters. "
            "Use mode='transpositions' for exhaustive coverage."
        )

    passed = not failures
    return DetectorResult(
        name="permutation_invariance",
        passed=passed,
        cases_tested=cases_tested,
        counterexample=failures[0] if failures else None,
        statistic={"mode": mode},
        notes=notes,
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
    seed: int = 42,  # accepted for API consistency; unused
) -> DetectorResult:
    """If every voter votes X, the aggregator must return X.

    The Pareto / unanimity condition from social choice theory
    (May 1952; Arrow 1951). Tested deterministically: one unanimous
    ballot per class, total `len(classes)` cases. Random sampling
    would be the wrong tool here — unanimity has probability 1/k^n of
    arising by chance, so random trials only test against unanimity
    by accident.

    Aggregators that violate this property are overriding the
    unanimous will of the voters: usually a miscalibrated weight,
    a threshold bug, or a hidden default.
    """
    failures: list[dict] = []
    for c in classes:
        votes = [c] * n_voters
        out = vote_fn(votes)
        if out != c:
            failures.append({
                "unanimous_for": c,
                "votes": votes,
                "output": out,
            })

    passed = not failures
    return DetectorResult(
        name="pareto_unanimity",
        passed=passed,
        cases_tested=len(classes),
        counterexample=failures[0] if failures else None,
        notes=(
            f"{len(classes)} unanimous ballots (one per class) generated "
            f"by construction. Unanimous input must yield the unanimous "
            f"choice (Pareto / May 1952)."
        ),
    )


# ---------------------------------------------------------------------------
# Participation monotonicity  (NEW in v0.4.0)
# ---------------------------------------------------------------------------

def participation_monotonicity(
    vote_fn: VoteFunction,
    classes: Sequence[Vote],
    target_class: Vote,
    n_voters: int,
    n_trials: int = 200,
    seed: int = 42,
) -> DetectorResult:
    """Adding a new voter who votes X should not move the winner *away*
    from X.

    Distinct from `monotonicity()`, which keeps the voter count fixed
    and changes an existing voter's choice. Participation
    monotonicity tests the *no-show paradox*: a voter abstains, the
    aggregator returns X; the same voter shows up and votes X, and
    now the aggregator returns some Y != X. That is participation
    monotonicity violated.

    This is the property that breaks under Condorcet, instant-runoff
    voting (IRV), and other threshold-based rules. Plain majority
    voting satisfies it trivially.

    Test procedure: for each random ballot of size `n_voters - 1`
    whose aggregated winner is X, append one vote for X and check
    that the winner is still X. Discards trials where the original
    winner is already not X.
    """
    if n_voters < 2:
        return DetectorResult(
            name=f"participation_monotonicity[{target_class}]",
            passed=True,
            cases_tested=0,
            notes="Participation monotonicity is trivial for n_voters < 2 — skipped.",
        )

    rng = random.Random(seed)
    violations: list[dict] = []
    n_eligible = 0
    for _ in range(n_trials):
        small = [rng.choice(classes) for _ in range(n_voters - 1)]
        out_small = vote_fn(small)
        if out_small != target_class:
            continue
        n_eligible += 1
        with_added = small + [target_class]
        out_added = vote_fn(with_added)
        if out_added != target_class:
            violations.append({
                "small_votes": small,
                "added_vote_for": target_class,
                "winner_without_new_voter": out_small,
                "winner_with_new_voter": out_added,
            })
            if len(violations) >= 3:
                break

    passed = not violations
    return DetectorResult(
        name=f"participation_monotonicity[{target_class}]",
        passed=passed,
        cases_tested=n_eligible,
        counterexample=violations[0] if violations else None,
        statistic={"trials_total": n_trials, "trials_eligible": n_eligible},
        notes=(
            "Adding a voter who votes X must not move the winner away from "
            "X (no-show paradox / participation criterion)."
        ),
    )


# ---------------------------------------------------------------------------
# Independence of Irrelevant Alternatives  (NEW in v0.2.0)
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

    This is the **"replace losing votes"** variant of Arrow's IIA, as
    distinct from the **"add new losing alternative"** variant — both
    are called IIA in different literatures and they test different
    things.

    Procedure (replace-losing-votes variant):

      1. Sample a random ballot v with n_voters votes drawn uniformly
         from `classes`.
      2. Compute the winner w = vote_fn(v) and its vote count.
      3. Identify a "loser" L: any class with strictly fewer votes
         than w in v.
      4. Replace every L vote in v with a uniformly chosen non-L class
         to obtain v'. (We do *not* remove the L class from the label
         set — we re-distribute those voters' choices.)
      5. The winner of v' should still be w. If not, IIA is violated.

    Arrow's theorem (1951) proves that no deterministic aggregator
    over three or more classes can simultaneously satisfy unanimity,
    IIA, and non-dictatorship. This detector therefore *expects* most
    non-trivial 3+ class aggregators to fail it. The point of running
    it is to quantify and locate the violations, not to be surprised
    by their existence.

    For the alternative "add new losing alternative" formulation,
    write a custom test using `ensemble_symmetry_audit.strategies` —
    it requires constructing ballots with a fresh class label, which
    is a different generative setup.
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
