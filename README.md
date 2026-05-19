# ensemble-bias-detector

Property-based bias detectors for voting ensembles.

A small, dependency-free Python library that audits any voting / majority
aggregator for the structural properties it ought to satisfy and surfaces
counterexamples when it does not.

## Why this exists

You can have eleven well-behaved voters and a clean weighted-majority
aggregator and still ship an ensemble that votes the same way 90%+ of
the time on inputs where it should be agnostic. The voters are
individually correct. The aggregator is mathematically correct. The
bias only appears in the joint distribution — exactly what unit tests
miss and property-based tests are designed to catch.

This library is a battery of six such property tests, packaged so you
can drop it into any project and audit your aggregator in a few lines.

## What it tests

Six properties every sensible voting aggregator should satisfy:

| # | Property                       | What it catches                                              |
|---|--------------------------------|--------------------------------------------------------------|
| 1 | Balanced-input symmetry        | A silent skew toward one class under uniform random input   |
| 2 | Regime-flip invariance         | Asymmetric reaction to mirror-image inputs                  |
| 3 | Null-majority abstention       | Picking sides when the evidence is even                     |
| 4 | Monotonicity                   | More votes for X paradoxically reducing X's chance of winning |
| 5 | Permutation invariance         | Output depending on voter order                             |
| 6 | Tie-break determinism          | Same input producing different outputs across runs          |

All detectors are pure Python, zero runtime dependencies.

## Install

```
pip install ensemble-bias-detector
```

Python 3.10+.

## Quick start

```python
from collections import Counter
from ensemble_bias_detector import audit


def my_ensemble(votes):
    return Counter(votes).most_common(1)[0][0]


report = audit(
    my_ensemble,
    classes=["UP", "DOWN"],
    n_voters=11,
    flip_map={"UP": "DOWN", "DOWN": "UP"},
)

print(report)
```

Output:

```
Ensemble bias audit report
----------------------------------------
[PASS] balanced_input_symmetry  (2000 cases)
[PASS] regime_flip_invariance   (500 cases)
[PASS] monotonicity[UP]         (200 cases)
[PASS] monotonicity[DOWN]       (200 cases)
[PASS] permutation_invariance   (200 cases)
[PASS] tie_break_determinism    (200 cases)
----------------------------------------
ALL PROPERTIES HELD
```

## Catching a phantom voter

```python
def biased_ensemble(votes):
    # two phantom DOWN voters silently appended on every call
    return Counter(list(votes) + ["DOWN", "DOWN"]).most_common(1)[0][0]


print(audit(biased_ensemble, classes=["UP", "DOWN"], n_voters=11,
            flip_map={"UP": "DOWN", "DOWN": "UP"}))
```

Output (abbreviated):

```
[FAIL] balanced_input_symmetry  (2000 cases)
         counterexample: {'observed_counts': {'UP': 576, 'DOWN': 1424},
         'max_relative_deviation': 0.424, ...}
[FAIL] regime_flip_invariance   (500 cases)
         counterexample: {'out_original': 'DOWN', 'out_flipped': 'DOWN',
         'expected_flipped': 'UP'}
```

The library does not just fail — it tells you which property failed
and shows you a concrete input that breaks it.

## Three or more classes: the harder case

Writing a deterministic aggregator that passes every property over
three or more classes is genuinely difficult. Three innocent-looking
3-class aggregators audited side by side:

```python
def naive(votes):
    # Counter insertion order silently picks tied winners
    return Counter(votes).most_common(1)[0][0]

def alphabetical(votes):
    counts = Counter(votes)
    top = max(counts.values())
    return sorted(c for c, k in counts.items() if k == top)[0]

def hash_break(votes):
    import hashlib
    counts = Counter(votes)
    top = max(counts.values())
    winners = sorted(c for c, k in counts.items() if k == top)
    if len(winners) == 1:
        return winners[0]
    key = ",".join(sorted(map(str, votes)))
    return winners[hashlib.md5(key.encode()).digest()[0] % len(winners)]
```

Running each through `audit(fn, ["A", "B", "C"], n_voters=11)`:

- **naive** passes the balance test but fails permutation invariance —
  its tie-break leaks the order voters arrived in.
- **alphabetical** is permutation-invariant but visibly favours A
  over B over C on uniform random input (around 28% deviation).
- **hash_break** is the closest to fair, but on three or more classes
  some residual bias is unavoidable in any *deterministic* tie-break.

See `examples/02_three_class_classifier.py` for the full run. The
larger point: voting aggregation has subtler failure modes than unit
tests reach, and the difficulty grows with the number of classes.

## Who this is for

- ML engineers shipping multi-classifier voting ensembles
- Quantitative researchers using model-vote decision rules
- Anyone building credit-scoring, triage, moderation, or recommender
  systems that aggregate several signals into a single choice
- Anyone who has ever stared at a confusion matrix wondering why one
  class wins almost every time

The asymmetry these tests catch is domain-agnostic. The same
detectors that audit a trading signal aggregator will audit a
medical-triage classifier or a recommender vote.

## Calling individual detectors

If you don't want the full battery, every detector is exported:

```python
from ensemble_bias_detector import balanced_input_symmetry

result = balanced_input_symmetry(
    my_ensemble,
    classes=["A", "B", "C"],
    n_voters=11,
    n_trials=2000,
    tolerance=0.10,
)
print(result)
```

Each detector returns a `DetectorResult` with `passed`, `cases_tested`,
`counterexample`, and `notes` fields.

## Integrating into pytest

```python
from ensemble_bias_detector import audit

def test_ensemble_is_unbiased():
    report = audit(my_ensemble, classes=["UP", "DOWN"],
                   n_voters=11,
                   flip_map={"UP": "DOWN", "DOWN": "UP"})
    assert report.all_passed, str(report)
```

## What the library does *not* do

- It does not test predictive accuracy. A perfectly unbiased aggregator
  can still be a bad model. This library only audits structural
  properties of the aggregation step.
- It does not assume a probabilistic interpretation of votes. If your
  votes carry weights or confidences, wrap your aggregator in a
  function that accepts a list of votes and returns a decision.
- It does not replace domain validation. Some aggregators are
  deliberately biased (e.g. a safety classifier that errs on the side
  of caution). The library flags asymmetry; you decide whether the
  asymmetry is intentional.

## Contributing

Issues and pull requests welcome. The project is small on purpose —
six detectors, no dependencies. Additions are considered if they
describe a property that's both broadly applicable and reasonably
likely to be violated in practice.

## License

MIT.
