"""Self-tests: detectors correctly identify known good and bad aggregators."""

import hashlib
import random
from collections import Counter

from ensemble_bias_detector.detectors import (
    balanced_input_symmetry,
    monotonicity,
    null_majority_abstention,
    permutation_invariance,
    regime_flip_invariance,
    tie_break_determinism,
)


CLASSES = ["A", "B", "C"]
FLIP = {"A": "B", "B": "A", "C": "C"}


# --- reference aggregators -------------------------------------------------

def fair_majority(votes):
    """Majority with a multiset-based hash tie-break.

    The tie-break must be permutation-invariant, so we hash the *sorted*
    multiset of votes rather than the raw sequence. This keeps the
    aggregator deterministic, label-symmetric on random input, and
    independent of voter order.
    """
    counts = Counter(votes)
    top = max(counts.values())
    winners = sorted(c for c, k in counts.items() if k == top)
    if len(winners) == 1:
        return winners[0]
    key = ",".join(sorted(map(str, votes)))
    digest = hashlib.md5(key.encode()).digest()
    return winners[digest[0] % len(winners)]


def biased_to_A(votes):
    counts = Counter(list(votes) + ["A", "A", "A", "A"])
    top = max(counts.values())
    winners = sorted(c for c, k in counts.items() if k == top)
    return winners[0]


def order_dependent(votes):
    return votes[0]


def random_tie_break(votes):
    counts = Counter(votes)
    top = max(counts.values())
    winners = [c for c, k in counts.items() if k == top]
    return random.choice(winners)  # non-deterministic on ties


def antitone(votes):
    """Adversarial: returns the class with the FEWEST votes."""
    counts = Counter(votes)
    for c in CLASSES:
        counts.setdefault(c, 0)
    bottom = min(counts.values())
    winners = sorted(c for c, k in counts.items() if k == bottom)
    return winners[0]


def never_neutral(votes):
    counts = Counter(votes)
    counts.pop("C", None)
    if not counts:
        return "A"
    top = max(counts.values())
    winners = sorted(c for c, k in counts.items() if k == top)
    return winners[0]


# --- balanced_input_symmetry -----------------------------------------------

def test_balanced_input_passes_for_fair():
    # With 2 labels and an odd number of voters, ties are impossible —
    # the aggregator reduces to a pure label-symmetric majority and the
    # output distribution is provably uniform on uniform input. With 3+
    # classes, every deterministic tie-break introduces some asymmetry,
    # which the detector also catches (see README for the discussion).
    binary_classes = ["UP", "DOWN"]
    r = balanced_input_symmetry(fair_majority, binary_classes, n_voters=11, seed=0)
    assert r.passed, r


def test_balanced_input_fails_for_biased():
    r = balanced_input_symmetry(biased_to_A, CLASSES, n_voters=11, seed=0)
    assert not r.passed
    assert r.counterexample is not None
    assert r.counterexample["observed_counts"]["A"] > r.counterexample["expected_per_class"]


# --- regime_flip_invariance ------------------------------------------------

def test_regime_flip_passes_for_fair():
    # Use a binary, fully-flippable label set with odd voters so ties are
    # impossible. Mixing a deterministic tie-break with a 3-class flip is
    # provably ambiguous (the multiset is flip-invariant but the picked
    # winner is not), so we keep the flip test on a setting that
    # admits a clean reference aggregator.
    binary_classes = ["UP", "DOWN"]
    binary_flip = {"UP": "DOWN", "DOWN": "UP"}
    r = regime_flip_invariance(fair_majority, binary_classes, binary_flip,
                                n_voters=11, seed=0)
    assert r.passed, r


def test_regime_flip_fails_for_biased():
    r = regime_flip_invariance(biased_to_A, CLASSES, FLIP, n_voters=11, seed=0)
    assert not r.passed
    assert r.counterexample is not None


# --- null_majority_abstention ---------------------------------------------

def test_null_majority_passes_for_fair():
    # With C as neutral, when A and B votes are balanced, fair_majority
    # falls back to alphabetical ordering on ties — which is fine: ties
    # are common, so the neutral rate should still be high.
    def neutral_on_tie(votes):
        counts = Counter(votes)
        # if non-neutral classes tie at top, pick neutral
        non_neutral_counts = {c: counts.get(c, 0) for c in CLASSES if c != "C"}
        if len(set(non_neutral_counts.values())) == 1:
            return "C"
        return fair_majority(votes)

    r = null_majority_abstention(
        neutral_on_tie, CLASSES, neutral_class="C", n_voters=10, seed=0
    )
    assert r.passed, r


def test_null_majority_fails_for_aggregator_that_never_picks_neutral():
    r = null_majority_abstention(
        never_neutral, CLASSES, neutral_class="C", n_voters=10, seed=0
    )
    assert not r.passed


# --- monotonicity ----------------------------------------------------------

def test_monotonicity_passes_for_fair():
    for target in CLASSES:
        r = monotonicity(fair_majority, CLASSES, target, n_voters=7, seed=0)
        assert r.passed, r


def test_monotonicity_fails_for_antitone():
    r = monotonicity(antitone, CLASSES, "A", n_voters=7, seed=0)
    assert not r.passed
    assert r.counterexample is not None


# --- permutation_invariance ------------------------------------------------

def test_permutation_passes_for_fair():
    r = permutation_invariance(fair_majority, CLASSES, n_voters=9, seed=0)
    assert r.passed, r


def test_permutation_fails_for_order_dependent():
    r = permutation_invariance(order_dependent, CLASSES, n_voters=9, seed=0)
    assert not r.passed
    assert r.counterexample is not None


# --- tie_break_determinism -------------------------------------------------

def test_tie_break_passes_for_fair():
    r = tie_break_determinism(fair_majority, CLASSES, n_voters=4, seed=0)
    assert r.passed, r


def test_tie_break_fails_for_random_tie_break():
    r = tie_break_determinism(random_tie_break, CLASSES, n_voters=4, seed=0)
    assert not r.passed
    assert r.counterexample is not None
