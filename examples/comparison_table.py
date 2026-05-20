"""Generate the README's at-a-glance comparison table.

Audits a small set of canonical aggregators with the same `(K, N)`
configuration and prints which properties each one passes. The result
is the table at the top of the README under
"Where does my aggregator stand?".

Run:

    python examples/comparison_table.py
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from ensemble_symmetry_audit import audit


# ---------------------------------------------------------------------------
# Aggregators under test
# ---------------------------------------------------------------------------

def plurality_hash_tie(votes):
    counts = Counter(votes)
    top = max(counts.values())
    winners = sorted(c for c, k in counts.items() if k == top)
    if len(winners) == 1:
        return winners[0]
    key = ",".join(sorted(map(str, votes)))
    return winners[hashlib.md5(key.encode()).digest()[0] % len(winners)]


def plurality_alpha_tie(votes):
    counts = Counter(votes)
    top = max(counts.values())
    return sorted(c for c, k in counts.items() if k == top)[0]


def insertion_order_tie(votes):
    return Counter(votes).most_common(1)[0][0]


def sklearn_hard(votes):
    # Reproduces sklearn.ensemble.VotingClassifier(voting='hard')
    arr = np.asarray(votes, dtype=int)
    return int(np.argmax(np.bincount(arr, minlength=3)))


def constant_aggregator(votes):
    # Always returns first class — pathological reference
    return 0 if isinstance(votes[0], int) else "A"


# ---------------------------------------------------------------------------
# Audit each one
# ---------------------------------------------------------------------------

def audit_aggregator(name, fn, classes, n_voters, flip_map=None):
    report = audit(fn, classes=classes, n_voters=n_voters,
                   flip_map=flip_map, seed=42)
    n_total = len(report.results)
    n_passed = sum(1 for r in report.results if r.passed)
    return {
        "name": name,
        "n_total": n_total,
        "n_passed": n_passed,
        "ratio": f"{n_passed}/{n_total}",
        "failed_props": [r.name for r in report.failed],
    }


def main():
    print("Generating comparison table on K=3 classes, N=5 voters")
    print("=" * 70)

    rows = [
        audit_aggregator(
            "Plurality with hash multiset tie-break",
            plurality_hash_tie, ["A", "B", "C"], n_voters=5,
        ),
        audit_aggregator(
            "Plurality with alphabetical tie-break",
            plurality_alpha_tie, ["A", "B", "C"], n_voters=5,
        ),
        audit_aggregator(
            "Counter.most_common() (insertion-order tie-break)",
            insertion_order_tie, ["A", "B", "C"], n_voters=5,
        ),
        audit_aggregator(
            "sklearn VotingClassifier(voting='hard')",
            sklearn_hard, [0, 1, 2], n_voters=5,
        ),
        audit_aggregator(
            "Constant aggregator (always returns first class)",
            constant_aggregator, ["A", "B", "C"], n_voters=5,
        ),
    ]

    print()
    print(f"{'Aggregator':<55} {'Passed':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['name']:<55} {r['ratio']:>10}")
    print()
    print("Failed properties per aggregator:")
    for r in rows:
        if r["failed_props"]:
            print(f"  {r['name']}:")
            for p in r["failed_props"]:
                print(f"    - {p}")
        else:
            print(f"  {r['name']}: (none)")

    # Markdown for README
    print()
    print("=" * 70)
    print("Markdown for README:")
    print("=" * 70)
    print()
    print("| Aggregator | Properties passed |")
    print("|---|---|")
    for r in rows:
        print(f"| {r['name']} | **{r['ratio']}** |")


if __name__ == "__main__":
    main()
