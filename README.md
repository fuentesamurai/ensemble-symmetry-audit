# ensemble-symmetry-audit

Property-based audit of voting ensemble aggregators.

A small Python library that audits any voting, majority, or weighted-vote
aggregator against eight structural properties from social-choice theory
and reports a minimal counterexample whenever one fails.

> **Note on the name.** *Symmetry* here means structural symmetry of the
> aggregation rule (Pareto, IIA, monotonicity, flip invariance, permutation
> invariance, tie-break determinism). This library does **not** audit
> demographic bias in the underlying classifiers — that is a separate
> problem with its own established tooling. See `Scope` below.

## Why this exists

You can have eleven well-behaved voters and a clean weighted-majority
aggregator and still ship an ensemble that picks the same class on 90%+
of inputs where it should be agnostic. Every voter is individually
unbiased. The aggregator is mathematically correct. The bias only
appears in the *joint distribution* of voters + aggregation rule —
which unit tests do not reach and property-based tests are designed to
expose.

This library is the audit suite I wish I had shipped with.

## What it tests

| # | Property                                          | What it catches                                                                                                     |
|---|---------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| 1 | **Pareto / unanimity** (May 1952)                 | Aggregators that override a unanimous vote (weight bugs, miscalibrated thresholds, hidden defaults)                |
| 2 | **Balanced-input symmetry** (chi-squared test)    | A silent skew toward one class under uniform random input                                                          |
| 3 | **Regime-flip invariance**                        | Asymmetric reaction to mirror-image inputs                                                                          |
| 4 | **Null-majority abstention** *(opt-in, binomial)* | Picking sides when the evidence is balanced — useful where abstention is permitted                                 |
| 5 | **Monotonicity**                                  | More votes for X paradoxically reducing X's chance of winning                                                       |
| 6 | **Permutation invariance**                        | Output depending on voter order                                                                                     |
| 7 | **Tie-break determinism**                         | Same input producing different outputs across runs                                                                  |
| 8 | **Independence of Irrelevant Alternatives** (Arrow 1951) | Adding or removing a losing class flipping the winner (Arrow's theorem proves no 3+ class rule satisfies this) |

Where statistical claims are made, the detector reports a formal test
result (chi-squared goodness-of-fit, one-sided binomial) with an
explicit significance level, so the audit conclusions are reproducible
across runs and defensible against reviewers.

## Install

```
pip install ensemble-symmetry-audit
```

Python 3.10+. Requires `scipy>=1.10` (used for `chisquare` and `binomtest`).

## Quick start

```python
from collections import Counter
from ensemble_symmetry_audit import audit


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

Output (abbreviated):

```
Ensemble symmetry audit report
--------------------------------------------------
[PASS] pareto_unanimity            (200 cases)
[PASS] balanced_input_symmetry     (2000 cases)
         statistic: test=chi-squared, chi2=0.072, p_value=0.7886, alpha=0.01, df=1
[PASS] regime_flip_invariance      (500 cases)
[PASS] monotonicity[UP]            (200 cases)
[PASS] monotonicity[DOWN]          (200 cases)
[PASS] permutation_invariance      (200 cases)
[PASS] tie_break_determinism       (200 cases)
[PASS] independence_of_irrelevant_alternatives  (0 cases — binary)
--------------------------------------------------
ALL PROPERTIES HELD
```

For CI integration:

```python
report.to_json()  # JSON-serialisable audit log
report.all_passed
report.failed     # list of failed DetectorResult objects
```

## Catching a phantom voter

```python
def biased_ensemble(votes):
    # two phantom DOWN voters silently appended on every call
    return Counter(list(votes) + ["DOWN", "DOWN"]).most_common(1)[0][0]


print(audit(biased_ensemble, classes=["UP", "DOWN"], n_voters=11,
            flip_map={"UP": "DOWN", "DOWN": "UP"}))
```

```
[FAIL] balanced_input_symmetry     (2000 cases)
         statistic: chi2=358.4, p_value=0.0000, alpha=0.01
         counterexample: {'observed_counts': {'UP': 576, 'DOWN': 1424}, ...}
[FAIL] regime_flip_invariance      (500 cases)
         counterexample: {'out_original': 'DOWN', 'out_flipped': 'DOWN',
                          'expected_flipped': 'UP'}
```

The report tells you which property failed, the test statistic that
detected it, and a minimal concrete input that reproduces the failure.

## Three or more classes: Arrow's theorem in practice

Writing a deterministic aggregator that passes every property over
three or more classes is **provably impossible** without giving up some
desirable structural property. Arrow's theorem (1951) proves that no
deterministic non-dictatorial rule over 3+ alternatives can simultaneously
satisfy unanimity (Pareto), independence of irrelevant alternatives
(IIA), and universal domain.

The library does not try to hide this. It quantifies *where* and *how
often* each property fails for your particular aggregator and class
count. `examples/02_three_class_classifier.py` audits three
innocent-looking three-class rules side by side:

- **`Counter.most_common`** passes balance but fails permutation
  invariance (insertion-order tie-break).
- **Alphabetical tie-break** passes permutation but fails balance
  (≈ 28% skew toward earlier-sorted labels).
- **Hash-based tie-break on the sorted multiset** passes both for
  uniform input but fails IIA — predictable, given Arrow's theorem.

`examples/03_sklearn_voting_classifier.py` does the same audit on
`sklearn.ensemble.VotingClassifier` — the hard-voting rule there is
`np.argmax(np.bincount(...))`, which silently favours lower-indexed
labels.

## Calling individual detectors

```python
from ensemble_symmetry_audit import balanced_input_symmetry

result = balanced_input_symmetry(
    my_ensemble,
    classes=["A", "B", "C"],
    n_voters=11,
    n_trials=5000,
    alpha=0.005,
    seed=42,
)
print(result)
```

Each detector returns a `DetectorResult` with `passed`, `cases_tested`,
`counterexample`, `statistic`, and `notes` fields. `result.to_dict()`
gives a machine-readable representation.

## Integrating into pytest

```python
from ensemble_symmetry_audit import audit

def test_ensemble_is_symmetric():
    report = audit(my_ensemble, classes=["UP", "DOWN"],
                   n_voters=11,
                   flip_map={"UP": "DOWN", "DOWN": "UP"})
    assert report.all_passed, str(report)
```

## Scope

**What this library audits:** the *aggregation function* of a voting
ensemble — the code that combines votes / scores / predictions into a
single decision.

**What it does *not* audit:**

- Predictive accuracy or generalisation of the underlying classifiers.
- Demographic bias (race, gender, age, geography) in the underlying
  classifiers or in their training data. That is a different problem
  with different tooling (`fairlearn`, `aif360`, etc.).
- Soft / probabilistic votes (List[Dict[class, prob]]). The current
  version handles categorical votes only; probabilistic support is on
  the roadmap.

If your aggregation step is non-trivial enough to warrant an audit,
this library is for you. If you need to audit demographic fairness or
classifier accuracy, use the established tools for those problems.

## Roadmap

- v0.3: Hypothesis integration (directed counterexample search +
  shrinking), soft-voting / probabilistic vote support.
- v0.4: Adapters for sklearn `VotingClassifier`, `StackingClassifier`,
  XGBoost / LightGBM ensembles.
- v0.5: CI reporters (JUnit XML, GitHub Actions annotations).

## Contributing

Issues and pull requests welcome. The project is small on purpose —
eight properties grounded in social-choice theory, no scope creep
toward classifier-level audits. Additions are considered if they
describe a property both broadly applicable and reasonably likely to
be violated in practice.

## References

- Kenneth J. Arrow, *Social Choice and Individual Values* (1951).
- Kenneth O. May, "A Set of Independent Necessary and Sufficient
  Conditions for Simple Majority Decision" (1952).
- John Hughes & Koen Claessen, "QuickCheck: A Lightweight Tool for
  Random Testing of Haskell Programs" (1999).

## License

MIT.
