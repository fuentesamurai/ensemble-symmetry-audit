"""Soft-voting (probabilistic) property detectors.

These mirror the hard-voting detectors but operate on probability
distributions per voter:

    vote_fn: List[Dict[class, prob]] -> class

Each detector samples random inputs from a numpy-seeded RNG, evaluates
the aggregator, and either runs a statistical test (balance) or
searches for a counterexample. For deeper directed search with
automatic shrinking, see `ensemble_symmetry_audit.hypothesis_search`
(requires the optional `[shrink]` extra).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import chisquare

from .detectors import DetectorResult

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
    simplex — for each voter, independent across voters. Reports
    chi-squared p-value alongside effect size (Cohen's w) so that
    statistically-significant-but-small deviations can be distinguished
    from structurally important ones.
    """
    import math
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
            "cohens_w": round(cohens_w, 3),
            "max_relative_deviation": round(max_rel_dev, 3),
        },
        notes=(
            "Dirichlet-uniform soft votes should yield uniform output "
            "(chi-squared with effect-size reporting)."
        ),
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

def _summed_probs(votes, classes):
    """Helper: return averaged probability vector over voters."""
    summed = np.zeros(len(classes))
    for v in votes:
        for i, c in enumerate(classes):
            summed[i] += v.get(c, 0.0)
    return summed / len(votes)


def soft_continuity(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    n_voters: int,
    n_trials: int = 200,
    epsilon: float = 1e-3,
    margin_threshold: float = 0.02,
    seed: int = 42,
) -> DetectorResult:
    """Far from the decision boundary, small perturbations should not
    flip the output.

    For each random ballot we measure the *decision margin*: the gap
    between the winning class's averaged probability and the second
    best. Cases with margin <= `margin_threshold` are tied (or
    near-tied) by construction — a perturbation flipping the output
    is mathematically expected there and is NOT a property violation.

    Cases with margin > `margin_threshold` are "robustly decided";
    perturbations of magnitude ε should not flip the output. The
    detector counts violations only among these robust cases.

    Output reports both the robust-case violation rate (the property
    that matters) and the share of inputs that landed near the
    boundary (informational, useful for choosing
    `margin_threshold` / `epsilon`).
    """
    rng = np.random.default_rng(seed)
    n_robust = 0
    n_robust_flips = 0
    n_near_boundary = 0
    n_near_boundary_flips = 0
    sample_violation: dict | None = None
    n_classes = len(classes)

    for _ in range(n_trials):
        votes = []
        for _ in range(n_voters):
            probs = rng.dirichlet(np.ones(n_classes))
            votes.append({c: float(p) for c, p in zip(classes, probs)})
        out_a = vote_fn(votes)
        avg = _summed_probs(votes, classes)
        sorted_avg = np.sort(avg)[::-1]
        margin = float(sorted_avg[0] - sorted_avg[1])

        idx = int(rng.integers(0, n_voters))
        noise = rng.normal(0.0, epsilon, size=n_classes)
        new_probs = np.array([votes[idx][c] for c in classes]) + noise
        new_probs = np.clip(new_probs, 1e-9, None)
        new_probs = new_probs / new_probs.sum()
        new_vote = {c: float(p) for c, p in zip(classes, new_probs)}
        new_votes = list(votes)
        new_votes[idx] = new_vote
        out_b = vote_fn(new_votes)
        flipped = out_a != out_b

        if margin > margin_threshold:
            n_robust += 1
            if flipped:
                n_robust_flips += 1
                if sample_violation is None:
                    sample_violation = {
                        "epsilon": epsilon,
                        "margin": round(margin, 4),
                        "perturbed_voter": idx,
                        "out_before": out_a,
                        "out_after": out_b,
                    }
        else:
            n_near_boundary += 1
            if flipped:
                n_near_boundary_flips += 1

    robust_rate = n_robust_flips / max(n_robust, 1)
    boundary_rate = n_near_boundary_flips / max(n_near_boundary, 1)
    # Allow up to 1% robust-case flips — near-zero is the textbook
    # expectation when margin > 2 epsilon.
    tolerance = 0.01
    passed = robust_rate <= tolerance and n_robust > 0
    return DetectorResult(
        name="soft_continuity",
        passed=passed,
        cases_tested=n_trials,
        counterexample=sample_violation if not passed else None,
        statistic={
            "epsilon": epsilon,
            "margin_threshold": margin_threshold,
            "robust_cases": n_robust,
            "robust_flip_rate": round(robust_rate, 4),
            "tolerance": tolerance,
            "near_boundary_cases": n_near_boundary,
            "near_boundary_flip_rate": round(boundary_rate, 4),
        },
        notes=(
            "Robust cases (margin > margin_threshold) should not flip "
            "under epsilon-magnitude perturbations. Near-boundary cases "
            "are reported separately — flips there are mathematically "
            "expected, not bugs."
        ),
    )


def soft_participation_monotonicity(
    vote_fn: SoftVoteFunction,
    classes: Sequence[Any],
    target_class: Any,
    n_voters: int,
    confidence: float = 0.9,
    n_trials: int = 200,
    seed: int = 42,
) -> DetectorResult:
    """Adding a new voter whose probability mass concentrates on X
    should not move the winner away from X.

    Soft-voting analogue of `participation_monotonicity`. The added
    voter's probability vector puts `confidence` mass on
    `target_class` and the remainder uniformly over the other classes.

    Catches the no-show paradox in probabilistic aggregators with
    non-linear combination rules.
    """
    if n_voters < 2:
        return DetectorResult(
            name=f"soft_participation_monotonicity[{target_class}]",
            passed=True,
            cases_tested=0,
            notes="Participation monotonicity is trivial for n_voters < 2 — skipped.",
        )

    rng = np.random.default_rng(seed)
    n_classes = len(classes)
    others = [c for c in classes if c != target_class]
    per_other = (1.0 - confidence) / max(len(others), 1)
    new_voter = {c: per_other for c in others}
    new_voter[target_class] = confidence

    violations: list[dict] = []
    n_eligible = 0
    for _ in range(n_trials):
        small = []
        for _ in range(n_voters - 1):
            probs = rng.dirichlet(np.ones(n_classes))
            small.append({c: float(p) for c, p in zip(classes, probs)})
        out_small = vote_fn(small)
        if out_small != target_class:
            continue
        n_eligible += 1
        out_added = vote_fn(small + [dict(new_voter)])
        if out_added != target_class:
            violations.append({
                "added_voter_confidence_on": target_class,
                "added_voter_confidence": confidence,
                "winner_without_new_voter": out_small,
                "winner_with_new_voter": out_added,
            })
            if len(violations) >= 3:
                break

    passed = not violations
    return DetectorResult(
        name=f"soft_participation_monotonicity[{target_class}]",
        passed=passed,
        cases_tested=n_eligible,
        counterexample=violations[0] if violations else None,
        statistic={"trials_total": n_trials, "trials_eligible": n_eligible},
        notes=(
            "Adding a high-confidence X voter must not move the winner "
            "away from X (soft no-show paradox)."
        ),
    )
