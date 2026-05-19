"""Tests for the high-level audit() function and AuditReport."""

import hashlib
import json
from collections import Counter

from ensemble_symmetry_audit import audit


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


def test_audit_runs_core_detectors():
    report = audit(fair_majority, CLASSES, n_voters=11,
                   neutral_class="HOLD", flip_map=FLIP)
    names = [r.name for r in report.results]
    assert "pareto_unanimity" in names
    assert "balanced_input_symmetry" in names
    assert "regime_flip_invariance" in names
    assert "permutation_invariance" in names
    assert "tie_break_determinism" in names
    assert "independence_of_irrelevant_alternatives" in names
    assert any(n.startswith("monotonicity") for n in names)
    assert any(n.startswith("participation_monotonicity") for n in names)


def test_audit_null_majority_is_opt_in():
    # By default null_majority_abstention is NOT included
    report = audit(fair_majority, CLASSES, n_voters=11,
                   neutral_class="HOLD", flip_map=FLIP)
    names = [r.name for r in report.results]
    assert "null_majority_abstention" not in names


def test_audit_null_majority_included_when_required():
    report = audit(fair_majority, CLASSES, n_voters=11,
                   neutral_class="HOLD", flip_map=FLIP,
                   require_abstention=True)
    names = [r.name for r in report.results]
    assert "null_majority_abstention" in names


def test_audit_require_abstention_needs_neutral_class():
    import pytest
    with pytest.raises(ValueError):
        audit(fair_majority, CLASSES, n_voters=11, require_abstention=True)


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


def test_report_str_contains_summary():
    report = audit(fair_majority, CLASSES, n_voters=11,
                   neutral_class="HOLD", flip_map=FLIP)
    s = str(report)
    assert "Ensemble symmetry audit report" in s
    assert "balanced_input_symmetry" in s


def test_report_to_json_is_parseable():
    report = audit(fair_majority, CLASSES, n_voters=11, flip_map=FLIP)
    s = report.to_json()
    parsed = json.loads(s)
    assert "results" in parsed
    assert "all_passed" in parsed
    assert "config" in parsed
    assert parsed["config"]["n_voters"] == 11
