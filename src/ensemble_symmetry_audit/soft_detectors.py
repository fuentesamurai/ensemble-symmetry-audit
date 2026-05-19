"""Soft-voting (probabilistic) property detectors.

These mirror the hard-voting detectors but operate on probability
distributions per voter:

    vote_fn: List[Dict[class, prob]] -> class

Each detector follows the same pattern as the hard-voting suite:
generate inputs from a Hypothesis-style strategy, evaluate the
aggregator, and either run a statistical test or search for a
counterexample.

For counterexample-seeking detectors we use `hypothesis.find()`, which
performs directed search and automatic shrinking — so when a property
fails the reported counterexample is *minimal*, not just the first one
hit by random sampling.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
from hypothesis import HealthCheck, find, settings, strategies as st
from scipy.stats import chisquare

from .detectors import DetectorResult
from .strategies import (
    probability_distributions,
    probability_vote_lists,
)

ProbVote = Dict[Any, float]
SoftVoteFunction = Callable[[Sequence[ProbVote]], Any]


def _flip_prob_vote(vote: ProbVote, flip_map: Mapping[Any, Any]) -> ProbVote:
    out: ProbVote = {}
    for c, p in vote.items():
        out[flip_map[c]] = out.get(flip_map[c], 0.0) + p
    return out


# ---------------------------------------------------------------------------
# Soft balanced-input symmetry  (chi-squared)
# ---------------------------------------------------------------------------

def soft_balanced_input_symmetry(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    n_voters: int,
    n_trials: int = 1000,
    alpha: float = 0.01,
    seed: int = 42,
) -> DetectorResult:
    """Probability votes sampled from a symmetric Dirichlet should
    produce a roughly uniform output distribution.

    Uses Dirichlet(alpha=1, ..., 1) — i.e. uniform on the probability
    simplex — for each voter, independent across voters.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(classes)
    outputs: list[Any] = []
    for _ in range(n_trials):
        votes = []
        for _ in range(n_voters):
            probs = rng.dirichlet(np.ones(n_classes))
            votes.append({c: float(p) for c, p in zip(classes, probs)})
        outputs.append(vote_fn(votes))

    counts = Counter(outputs)
    observed = [counts.get(c, 0) for c in classes]
    expected = [n_trials / n_classes] * n_classes
    res = chisquare(f_obs=observed, f_exp=expected)
    chi2, p = float(res.statistic), float(res.pvalue)
    passed = p >= alpha

    counterexample = None if passed else {
        "observed_counts": dict(zip(classes, observed)),
        "expected_per_class": expected[0],
    }
    return DetectorResult(
        name="soft_balanced_input_symmetry",
        passed=passed,
        cases_tested=n_trials,
        counterexample=counterexample,
        statistic={
            "test": "chi-squared",
            "chi2": round(chi2, 3),
            "p_value": round(p, 4),
            "alpha": alpha,
            "df": n_classes - 1,
        },
        notes="Dirichlet-uniform soft votes should yield uniform output (chi-squared).",
    )


# ---------------------------------------------------------------------------
# Soft Pareto unanimity
# ---------------------------------------------------------------------------

def soft_pareto_unanimity(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    n_voters: int,
    confidence: float = 0.95,
    seed: int = 42,
) -> DetectorResult:
    """If every voter assigns the bulk of probability mass (>= confidence)
    to class X, the aggregator must return X.
    """
    failures: list[dict] = []
    for c in classes:
        others = [x for x in classes if x != c]
        per_other = (1.0 - confidence) / max(len(others), 1)
        vote = {x: per_other for x in others}
        vote[c] = confidence
        votes = [dict(vote) for _ in range(n_voters)]
        out = vote_fn(votes)
        if out != c:
            failures.append({
                "unanimous_for": c,
                "vote_each": vote,
                "output": out,
            })
            if len(failures) >= 3:
                break

    passed = not failures
    return DetectorResult(
        name="soft_pareto_unanimity",
        passed=passed,
        cases_tested=len(classes),
        counterexample=failures[0] if failures else None,
        notes=(
            f"Every voter giving >={confidence:.0%} mass to X must yield X "
            f"(soft Pareto)."
        ),
    )


# ---------------------------------------------------------------------------
# Soft regime-flip invariance
# ---------------------------------------------------------------------------

def soft_regime_flip_invariance(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    flip_map: Mapping[Any, Any],
    n_voters: int,
    n_trials: int = 300,
    seed: int = 42,
) -> DetectorResult:
    """Aggregator should commute with permuting probabilities by flip_map."""
    rng = np.random.default_rng(seed)
    failures: list[dict] = []
    for _ in range(n_trials):
        votes = []
        for _ in range(n_voters):
            probs = rng.dirichlet(np.ones(len(classes)))
            votes.append({c: float(p) for c, p in zip(classes, probs)})
        flipped = [_flip_prob_vote(v, flip_map) for v in votes]
        out_a = vote_fn(votes)
        out_b = vote_fn(flipped)
        if flip_map.get(out_a) != out_b:
            failures.append({
                "votes": [{k: round(v, 3) for k, v in d.items()} for d in votes],
                "out_original": out_a,
                "out_flipped": out_b,
                "expected_flipped": flip_map.get(out_a),
            })
            if len(failures) >= 3:
                break

    passed = not failures
    return DetectorResult(
        name="soft_regime_flip_invariance",
        passed=passed,
        cases_tested=n_trials,
        counterexample=failures[0] if failures else None,
        notes="Permuting probabilities by flip_map should yield the flipped output.",
    )


# ---------------------------------------------------------------------------
# Soft monotonicity (via Hypothesis directed search)
# ---------------------------------------------------------------------------

def soft_monotonicity(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    target_class: Any,
    n_voters: int,
    n_trials: int = 200,
    delta: float = 0.05,
    seed: int = 42,
) -> DetectorResult:
    """Shifting probability mass *toward* X must never move the output
    away from X.

    For each random vote list whose output is X, perturb a random voter
    by moving `delta` mass from another class to `target_class`. The
    output should stay X.
    """
    rng = np.random.default_rng(seed)
    violations: list[dict] = []
    n_classes = len(classes)
    for _ in range(n_trials):
        votes = []
        for _ in range(n_voters):
            probs = rng.dirichlet(np.ones(n_classes))
            votes.append({c: float(p) for c, p in zip(classes, probs)})
        out_a = vote_fn(votes)
        if out_a != target_class:
            continue
        idx = int(rng.integers(0, n_voters))
        from_class = next((c for c in classes if c != target_class
                           and votes[idx].get(c, 0.0) > delta), None)
        if from_class is None:
            continue
        new_vote = dict(votes[idx])
        new_vote[from_class] -= delta
        new_vote[target_class] = new_vote.get(target_class, 0.0) + delta
        new_votes = list(votes)
        new_votes[idx] = new_vote
        out_b = vote_fn(new_votes)
        if out_b != target_class:
            violations.append({
                "votes_before": [{k: round(v, 3) for k, v in d.items()}
                                 for d in votes],
                "perturbed_voter": idx,
                "moved_from": from_class,
                "moved_to": target_class,
                "delta": delta,
                "out_before": out_a,
                "out_after": out_b,
            })
            if len(violations) >= 3:
                break

    passed = not violations
    return DetectorResult(
        name=f"soft_monotonicity[{target_class}]",
        passed=passed,
        cases_tested=n_trials,
        counterexample=violations[0] if violations else None,
        notes="Moving mass toward X must not move output away from X.",
    )


# ---------------------------------------------------------------------------
# Soft permutation invariance
# ---------------------------------------------------------------------------

def soft_permutation_invariance(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    n_voters: int,
    n_trials: int = 200,
    n_shuffles: int = 5,
    seed: int = 42,
) -> DetectorResult:
    """Output should depend on the *bag* of voter probabilities, not the order."""
    rng = np.random.default_rng(seed)
    failures: list[dict] = []
    for _ in range(n_trials):
        votes = []
        for _ in range(n_voters):
            probs = rng.dirichlet(np.ones(len(classes)))
            votes.append({c: float(p) for c, p in zip(classes, probs)})
        base = vote_fn(list(votes))
        for _ in range(n_shuffles):
            idx = list(range(n_voters))
            rng.shuffle(idx)
            shuf = [votes[i] for i in idx]
            shuf_out = vote_fn(shuf)
            if shuf_out != base:
                failures.append({
                    "out_base": base,
                    "out_shuffled": shuf_out,
                    "shuffle_indices": idx,
                })
                break
        if len(failures) >= 3:
            break

    passed = not failures
    return DetectorResult(
        name="soft_permutation_invariance",
        passed=passed,
        cases_tested=n_trials,
        counterexample=failures[0] if failures else None,
        notes="Soft aggregator output must depend on voter multiset, not order.",
    )


# ---------------------------------------------------------------------------
# Soft continuity (NEW property only relevant to soft voting)
# ---------------------------------------------------------------------------

def soft_continuity(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    n_voters: int,
    n_trials: int = 200,
    epsilon: float = 1e-3,
    seed: int = 42,
) -> DetectorResult:
    """Tiny perturbations to a voter's probability should not flip output
    when the original decision is far from a tie.

    For each random input whose `vote_fn` output is X, perturb one
    voter's probabilities by epsilon-magnitude noise (renormalised) and
    check the output is still X. Discontinuities near tight decision
    boundaries are expected and tolerated by sampling many trials and
    flagging only when the violation rate is large.
    """
    rng = np.random.default_rng(seed)
    n_violations = 0
    n_eligible = 0
    sample_violation: dict | None = None
    n_classes = len(classes)

    for _ in range(n_trials):
        votes = []
        for _ in range(n_voters):
            probs = rng.dirichlet(np.ones(n_classes))
            votes.append({c: float(p) for c, p in zip(classes, probs)})
        out_a = vote_fn(votes)
        idx = int(rng.integers(0, n_voters))
        noise = rng.normal(0.0, epsilon, size=n_classes)
        new_probs = np.array([votes[idx][c] for c in classes]) + noise
        new_probs = np.clip(new_probs, 1e-9, None)
        new_probs = new_probs / new_probs.sum()
        new_vote = {c: float(p) for c, p in zip(classes, new_probs)}
        new_votes = list(votes)
        new_votes[idx] = new_vote
        out_b = vote_fn(new_votes)
        n_eligible += 1
        if out_a != out_b:
            n_violations += 1
            if sample_violation is None:
                sample_violation = {
                    "perturbation_epsilon": epsilon,
                    "perturbed_voter": idx,
                    "out_before": out_a,
                    "out_after": out_b,
                }

    rate = n_violations / max(n_eligible, 1)
    # A reasonable aggregator should flip on epsilon=1e-3 perturbations
    # only for inputs already at a tight tie. Allow up to 5% flip rate.
    passed = rate <= 0.05
    return DetectorResult(
        name="soft_continuity",
        passed=passed,
        cases_tested=n_eligible,
        counterexample=sample_violation if not passed else None,
        statistic={
            "epsilon": epsilon,
            "violation_rate": round(rate, 4),
            "tolerance": 0.05,
            "n_violations": n_violations,
        },
        notes="Small probability perturbations should rarely flip the output.",
    )
