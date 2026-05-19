"""Example 1: catching a 'phantom voter' bias.

Two aggregators over a binary UP / DOWN label:
  - fair_majority: plain majority with 11 voters (odd, so no ties)
  - biased_majority: same, but with two DOWN votes silently appended
    on every call (a 'phantom voter')

The fair aggregator passes every property. The biased one fails
exactly the properties designed to catch a hidden directional skew.
"""

from collections import Counter
from ensemble_symmetry_audit import audit


CLASSES = ["UP", "DOWN"]
FLIP = {"UP": "DOWN", "DOWN": "UP"}


def fair_majority(votes):
    return Counter(votes).most_common(1)[0][0]


def biased_majority(votes):
    return Counter(list(votes) + ["DOWN", "DOWN"]).most_common(1)[0][0]


def main():
    print("=" * 60)
    print("FAIR aggregator (pure majority over 11 voters)")
    print("=" * 60)
    print(audit(fair_majority, CLASSES, n_voters=11, flip_map=FLIP))

    print()
    print("=" * 60)
    print("BIASED aggregator (two phantom DOWN voters silently injected)")
    print("=" * 60)
    print(audit(biased_majority, CLASSES, n_voters=11, flip_map=FLIP))


if __name__ == "__main__":
    main()
