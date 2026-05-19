"""Case study: auditing scikit-learn's voting ensembles for structural bias.

We wrap the aggregation step of sklearn's most-used voting ensembles
(VotingClassifier, BaggingClassifier, RandomForestClassifier,
ExtraTreesClassifier) and run them through the hard and soft audit
suites across a grid of (n_classes, n_voters) configurations.

The aim is to quantify positional bias introduced by `np.argmax`'s
left-favouring tie-break — a behaviour present in every sklearn
ensemble that ends in `argmax`. The bias is invisible in unit tests
and absent from the API documentation.

Run:

    python examples/case_study_sklearn.py

Output:
    - Console: full audit reports and a summary table
    - case_study_sklearn_results.json: machine-readable findings
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ensemble_symmetry_audit import audit, soft_audit


# ---------------------------------------------------------------------------
# Aggregator wrappers — exact reproductions of sklearn's aggregation rules
# ---------------------------------------------------------------------------

def make_voting_hard_aggregator(n_classes: int):
    """Reproduces sklearn.ensemble.VotingClassifier(voting='hard').

    Sklearn source: np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x, weights=...)),
        axis=1, arr=predictions
    )

    Same rule is used by BaggingClassifier(voting='hard') and
    (with weights=None) by RandomForestClassifier when there is no
    predict_proba — i.e. the hard-vote internal fall-back.
    """
    def aggregate(votes):
        arr = np.asarray(votes, dtype=int)
        return int(np.argmax(np.bincount(arr, minlength=n_classes)))
    return aggregate


def make_voting_soft_aggregator(classes):
    """Reproduces sklearn.ensemble.VotingClassifier(voting='soft') /
    RandomForestClassifier.predict() / ExtraTreesClassifier.predict() /
    BaggingClassifier(voting='soft').

    All four average predict_proba outputs across base estimators and
    then take np.argmax. The argmax tie-break is positional (favours
    the lower index), same as in hard voting.
    """
    def aggregate(votes):
        n = len(classes)
        summed = np.zeros(n)
        for v in votes:
            for i, c in enumerate(classes):
                summed[i] += v.get(c, 0.0)
        return classes[int(np.argmax(summed))]
    return aggregate


# ---------------------------------------------------------------------------
# Grid audit
# ---------------------------------------------------------------------------

HARD_TARGETS = [
    "VotingClassifier(voting='hard')",
    "BaggingClassifier(voting='hard')",
    "RandomForestClassifier (hard-vote fallback)",
]

SOFT_TARGETS = [
    "VotingClassifier(voting='soft')",
    "RandomForestClassifier.predict()",
    "ExtraTreesClassifier.predict()",
    "BaggingClassifier(voting='soft')",
]


def grid_specs():
    """All (n_classes, n_voters) combinations to test."""
    return [
        (K, N) for K in [3, 5, 10] for N in [3, 5, 7, 11]
    ]


def audit_hard_grid():
    rows = []
    for K, N in grid_specs():
        classes = list(range(K))
        agg = make_voting_hard_aggregator(K)
        report = audit(agg, classes=classes, n_voters=N, seed=42)
        balanced = next(r for r in report.results
                        if r.name == "balanced_input_symmetry")
        observed = (balanced.counterexample or {}).get("observed_counts",
                                                       {c: 0 for c in classes})
        total = sum(observed.values()) or 1
        share_0 = observed.get(0, 0) / total
        expected = 1.0 / K
        relative_bias = (share_0 - expected) / expected
        rows.append({
            "ensemble": "sklearn hard voting (VotingClassifier / Bagging hard)",
            "n_classes": K,
            "n_voters": N,
            "balanced_passed": balanced.passed,
            "p_value": balanced.statistic["p_value"],
            "chi2": balanced.statistic["chi2"],
            "observed_share_class_0": round(share_0, 3),
            "expected_share": round(expected, 3),
            "relative_bias_class_0": round(relative_bias, 3),
            "failed_properties": [r.name for r in report.failed],
        })
    return rows


def audit_soft_grid():
    rows = []
    for K, N in grid_specs():
        classes = list(range(K))
        agg = make_voting_soft_aggregator(classes)
        report = soft_audit(agg, classes=classes, n_voters=N, seed=42)
        balanced = next(r for r in report.results
                        if r.name == "soft_balanced_input_symmetry")
        observed = (balanced.counterexample or {}).get("observed_counts",
                                                       {c: 0 for c in classes})
        total = sum(observed.values()) or 1
        share_0 = observed.get(0, 0) / total
        expected = 1.0 / K
        relative_bias = (share_0 - expected) / expected
        rows.append({
            "ensemble": "sklearn soft voting (VotingClassifier soft / RF / ET / Bagging soft)",
            "n_classes": K,
            "n_voters": N,
            "balanced_passed": balanced.passed,
            "p_value": balanced.statistic["p_value"],
            "chi2": balanced.statistic["chi2"],
            "observed_share_class_0": round(share_0, 3),
            "expected_share": round(expected, 3),
            "relative_bias_class_0": round(relative_bias, 3),
            "failed_properties": [r.name for r in report.failed],
        })
    return rows


def print_table(rows, title):
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)
    print(f"{'K':>3} {'N':>3} | {'pass':>5} | {'p_value':>9} | "
          f"{'chi2':>9} | {'share_c0':>9} | {'expected':>9} | {'rel_bias':>9}")
    print("-" * 90)
    for r in rows:
        flag = "PASS" if r["balanced_passed"] else "FAIL"
        print(f"{r['n_classes']:>3} {r['n_voters']:>3} | {flag:>5} | "
              f"{r['p_value']:>9.4f} | {r['chi2']:>9.2f} | "
              f"{r['observed_share_class_0']:>9.3f} | "
              f"{r['expected_share']:>9.3f} | "
              f"{r['relative_bias_class_0']:>+9.3f}")
    print()


def main():
    print("Auditing scikit-learn voting ensembles for structural positional bias")
    print("Library: ensemble-symmetry-audit v0.3.0")
    print()
    print("Hard voting targets (every ensemble below shares one aggregation rule):")
    for t in HARD_TARGETS:
        print(f"  - {t}")
    print()
    print("Soft voting targets (every ensemble below shares one aggregation rule):")
    for t in SOFT_TARGETS:
        print(f"  - {t}")
    print()

    hard_rows = audit_hard_grid()
    soft_rows = audit_soft_grid()

    print_table(hard_rows,
                "HARD voting — np.argmax(np.bincount(predictions))")
    print_table(soft_rows,
                "SOFT voting — np.argmax(mean(predict_proba))")

    out_path = Path(__file__).parent / "case_study_sklearn_results.json"
    with out_path.open("w") as f:
        json.dump({
            "library": "ensemble-symmetry-audit",
            "version": "0.3.0",
            "alpha_level": 0.01,
            "hard": hard_rows,
            "soft": soft_rows,
        }, f, indent=2)
    print(f"Machine-readable results: {out_path}")

    # Headline numbers
    print("\n" + "=" * 90)
    print("HEADLINE FINDINGS")
    print("=" * 90)
    hard_failures = [r for r in hard_rows if not r["balanced_passed"]]
    soft_failures = [r for r in soft_rows if not r["balanced_passed"]]
    worst_hard = max(hard_rows, key=lambda r: r["relative_bias_class_0"])
    worst_soft = max(soft_rows, key=lambda r: r["relative_bias_class_0"])

    print(f"Hard voting: {len(hard_failures)}/{len(hard_rows)} configurations "
          f"fail balanced_input_symmetry at alpha=0.01")
    print(f"  Worst case: K={worst_hard['n_classes']} classes, "
          f"N={worst_hard['n_voters']} voters")
    print(f"    Class 0 wins {worst_hard['observed_share_class_0']:.1%} of "
          f"trials, expected {worst_hard['expected_share']:.1%}")
    print(f"    Relative bias toward class 0: "
          f"{worst_hard['relative_bias_class_0']:+.1%}")

    print()
    print(f"Soft voting: {len(soft_failures)}/{len(soft_rows)} configurations "
          f"fail balanced_input_symmetry at alpha=0.01")
    print(f"  Worst case: K={worst_soft['n_classes']} classes, "
          f"N={worst_soft['n_voters']} voters")
    print(f"    Class 0 wins {worst_soft['observed_share_class_0']:.1%} of "
          f"trials, expected {worst_soft['expected_share']:.1%}")
    print(f"    Relative bias toward class 0: "
          f"{worst_soft['relative_bias_class_0']:+.1%}")


if __name__ == "__main__":
    main()
