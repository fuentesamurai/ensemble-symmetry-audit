"""Tests for the high-level audit() function and AuditReport."""

import hashlib
from collections import Counter

from ensemble_bias_detector import audit


CLASSES = ["BUY", "SELL", "HOLD"]
FLIP = {"BUY": "SELL", "SELL": "BUY", "HOLD": "HOLD"}


def fair_majority(votes):
    counts = Counter(votes)
    top = max(counts.values())
    winners = sorted(c for c, k in counts.items() if k == top)
    if len(winners) == 1:
        return winners[0]
    key = ",".join(sorted(map(str, votes)))
    digest = hashlib.md5(key.encode()).digest()
    return winners[digest[0] % len(winners)]


def biased_to_sell(votes):
    counts = Counter(list(votes) + ["SELL", "SELL", "SELL", "SELL"])
    top = max(counts.values())
    winners = sorted(c for c, k in counts.items() if k == top)
    return winners[0]


def test_audit_runs_all_detectors():
    report = audit(fair_majority, CLASSES, n_voters=11,
                   neutral_class="HOLD", flip_map=FLIP)
    names = [r.name for r in report.results]
    assert "balanced_input_symmetry" in names
    assert "regime_flip_invariance" in names
    assert "null_majority_abstention" in names
    assert "permutation_invariance" in names
    assert "tie_break_determinism" in names
    assert any(n.startswith("monotonicity") for n in names)


def test_audit_detects_directional_bias():
    report = audit(biased_to_sell, CLASSES, n_voters=11,
                   neutral_class="HOLD", flip_map=FLIP)
    assert not report.all_passed
    assert any(r.name == "balanced_input_symmetry" and not r.passed
               for r in report.results)


def test_audit_skips_flip_when_no_flip_map():
    report = audit(fair_majority, CLASSES, n_voters=11,
                   neutral_class="HOLD")
    names = [r.name for r in report.results]
    assert "regime_flip_invariance" not in names


def test_audit_skips_neutral_when_no_neutral_class():
    report = audit(fair_majority, CLASSES, n_voters=11)
    names = [r.name for r in report.results]
    assert "null_majority_abstention" not in names


def test_report_str_contains_summary():
    report = audit(fair_majority, CLASSES, n_voters=11,
                   neutral_class="HOLD", flip_map=FLIP)
    s = str(report)
    assert "Ensemble bias audit report" in s
    assert "balanced_input_symmetry" in s
