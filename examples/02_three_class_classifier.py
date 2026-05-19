"""Example 2: subtle 3-class bias from 'innocent' code.

Three voting aggregators that all look reasonable but have very
different bias profiles when audited:

  - naive_majority:  Counter(votes).most_common(1)[0][0]
                     Looks innocent. Insertion-order tie-break means
                     classes get a positional advantage you didn't
                     consciously choose.
  - alphabetical:    sort tied winners, return the first.
                     Looks deterministic. Silently favours labels
                     that sort earlier — a hidden alphabetical bias.
  - hash_break:      hash the sorted multiset, modulo number of winners.
                     Distributes ties pseudo-randomly. Still not
                     perfectly uniform on 3 classes, but the closest
                     of the three.

The point is: writing a *truly* unbiased deterministic aggregator over
three or more classes is harder than it looks. Run this example and
read the audit reports — every aggregator has a story to tell.
"""

import hashlib
from collections import Counter
from ensemble_symmetry_audit import audit


CLASSES = ["A", "B", "C"]


def naive_majority(votes):
    return Counter(votes).most_common(1)[0][0]


def alphabetical(votes):
    counts = Counter(votes)
    top = max(counts.values())
    return sorted(c for c, k in counts.items() if k == top)[0]


def hash_break(votes):
    counts = Counter(votes)
    top = max(counts.values())
    winners = sorted(c for c, k in counts.items() if k == top)
    if len(winners) == 1:
        return winners[0]
    key = ",".join(sorted(map(str, votes)))
    digest = hashlib.md5(key.encode()).digest()
    return winners[digest[0] % len(winners)]


def main():
    for name, fn in [("naive_majority", naive_majority),
                     ("alphabetical",   alphabetical),
                     ("hash_break",     hash_break)]:
        print("=" * 60)
        print(f"Auditing {name}")
        print("=" * 60)
        print(audit(fn, CLASSES, n_voters=11))
        print()


if __name__ == "__main__":
    main()
