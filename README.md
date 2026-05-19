# ensemble-symmetry-audit

Property-based audit of voting ensemble aggregators.

A small Python library that audits any voting, majority, or
weighted-vote aggregator against nine structural properties from
social-choice theory and reports a minimal counterexample whenever one
fails. Hard *and* soft (probabilistic) voting are both supported.

> **Note on the name.** *Symmetry* here means structural symmetry of
> the aggregation rule (Pareto, IIA, monotonicity, participation,
> flip invariance, permutation invariance, tie-break determinism). This
> library does **not** audit demographic bias in the underlying
> classifiers — that is a separate problem with its own established
> tooling (see `Scope`).

## The phantom voter

You can have eleven well-behaved voters and a clean weighted-majority
aggregator and still ship an ensemble that picks the same class on 90%+
of inputs where it should be agnostic.

```python
from collections import Counter
from ensemble_symmetry_audit import audit


def biased_ensemble(votes):
    # Two phantom DOWN voters silently appended on every call
    return Counter(list(votes) + ["DOWN", "DOWN"]).most_common(1)[0][0]


print(audit(biased_ensemble, classes=["UP", "DOWN"], n_voters=11,
            flip_map={"UP": "DOWN", "DOWN": "UP"}))
```

Output (abbreviated):

```
[FAIL] balanced_input_symmetry     (2000 cases)
         statistic: chi2=369.8, p_value=0.0, alpha=0.01
         counterexample: {'observed_counts': {'UP': 570, 'DOWN': 1430}}
[FAIL] regime_flip_invariance      (500 cases)
         counterexample: {'out_original': 'DOWN', 'out_flipped': 'DOWN',
                          'expected_flipped': 'UP'}
```

The report tells you which property failed, the test statistic that
detected it, and a minimal concrete input that reproduces the failure.

## Install

```
pip install ensemble-symmetry-audit
```

Python 3.10+. Requires `scipy>=1.10` and `numpy>=1.20`.

For optional Hypothesis-driven counterexample shrinking:

```
pip install ensemble-symmetry-audit[shrink]
```

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

print(report)            # human-readable report
report.all_passed        # bool
report.failed            # list of failing DetectorResults
report.to_json()         # JSON for CI integration
```

## What it tests

| #  | Property                                                | What it catches                                                                              |
|----|---------------------------------------------------------|----------------------------------------------------------------------------------------------|
| 1  | **Pareto / unanimity** (May 1952)                       | Aggregators that override a unanimous vote (weight bugs, miscalibrated thresholds, defaults) |
| 2  | **Balanced-input symmetry** (chi-squared test)          | A silent skew toward one class under uniform random input                                    |
| 3  | **Regime-flip invariance**                              | Asymmetric reaction to mirror-image inputs                                                   |
| 4  | **Null-majority abstention** *(opt-in, binomial)*       | Picking sides when the evidence is balanced — where abstention is permitted                  |
| 5  | **Monotonicity**                                        | More votes for X paradoxically reducing X's chance of winning                                |
| 6  | **Participation monotonicity** *(new v0.4)*             | The no-show paradox: adding an X-voter making X lose                                         |
| 7  | **Permutation invariance** *(adjacent-transpositions)*  | Output depending on voter order — exhaustive coverage via the generators of S_n              |
| 8  | **Tie-break determinism**                               | Same input producing different outputs across runs                                           |
| 9  | **Independence of Irrelevant Alternatives** (Arrow 1951) | Adding or removing a losing class flipping the winner                                        |

Statistical claims use formal hypothesis tests (chi-squared
goodness-of-fit, one-sided binomial) with explicit significance levels.

The same properties (plus two soft-only ones) are also available for
soft / probabilistic aggregators via `soft_audit()`. See
[docs/soft-voting.md](docs/soft-voting.md).

## Three or more classes: Arrow's theorem in practice

Writing a deterministic aggregator that passes every property over
three or more classes is **provably impossible** without giving up
some desirable structural property. Arrow's theorem (1951) proves
that no deterministic non-dictatorial rule over 3+ alternatives can
simultaneously satisfy unanimity (Pareto), independence of irrelevant
alternatives (IIA), and universal domain.

The library does not try to hide this. It quantifies *where* and
*how often* each property fails for your particular aggregator and
class count. `examples/02_three_class_classifier.py` audits three
innocent-looking three-class rules side by side:

- `Counter.most_common` passes balance but fails permutation invariance
  (insertion-order tie-break).
- Alphabetical tie-break passes permutation but fails balance
  (≈ 28% skew toward earlier-sorted labels).
- Hash-based tie-break on the sorted multiset passes both for uniform
  input but fails IIA — predictable, given Arrow's theorem.

## Case study: auditing scikit-learn's voting ensembles

`examples/case_study_sklearn.py` maps the bias quantitatively across
twelve `(n_classes, n_voters)` configurations of sklearn's hard- and
soft-voting ensembles. Headline findings:

- **Hard voting** (`VotingClassifier(voting='hard')`,
  `BaggingClassifier`, `RandomForestClassifier` hard-vote fallback)
  **fails balanced-input symmetry in all 12 configurations** at
  α=0.01.
- The worst case is K=10 classes, N=3 voters: class 0 wins
  **23.8% of decisions instead of the expected 10%** — a +138%
  relative advantage from the positional `np.argmax` tie-break alone.
- **Soft voting** passes the same test in **11 of 12 configurations**:
  real-valued probability averaging almost never produces exact ties
  for the tie-break to bite.

Full grid output, including chi-squared statistics and p-values, is
in `examples/case_study_sklearn_results.json`.

## Documentation

- [docs/soft-voting.md](docs/soft-voting.md) — soft-voting suite,
  `soft_continuity` and the boundary problem
- [docs/hypothesis-shrinking.md](docs/hypothesis-shrinking.md) —
  optional Hypothesis-driven counterexample shrinking and strategies
- [docs/references.md](docs/references.md) — Arrow, May, QuickCheck,
  and related tooling

## Scope

**What this library audits:** the *aggregation function* of a voting
ensemble — the code that combines votes / scores / predictions into a
single decision.

**What it does *not* audit:**

- Predictive accuracy or generalisation of the underlying classifiers.
- Demographic bias (race, gender, age, geography) in the underlying
  classifiers or in their training data. That is a different problem
  with different tooling (`fairlearn`, `aif360`, etc.).
- Calibration of the *probabilities* produced by soft-voting
  ensembles. The library tests structural properties of the
  aggregation function, not whether the resulting probabilities are
  well-calibrated against empirical frequencies.

## Roadmap

**Latest — v0.4.0** (May 2026):
- `participation_monotonicity` detector (hard + soft) — catches the
  no-show paradox.
- `soft_continuity` is now boundary-aware (no more false positives
  near tight ties).
- `permutation_invariance` defaults to exhaustive adjacent
  transpositions.
- Hypothesis moved to the optional `[shrink]` extra.
- README split into a focused top-level file plus `docs/`.

**Next:**
- **v0.5** — first-class adapters for sklearn `VotingClassifier`,
  `StackingClassifier`, XGBoost / LightGBM ensembles.
- **v0.6** — CI reporters (JUnit XML, GitHub Actions annotations) and
  HTML / Markdown report exporters.
- **v0.7** — soft-vote calibration property tests.

See [Releases](https://github.com/fuentesamurai/ensemble-symmetry-audit/releases)
for the full version history.

## Contributing

Issues and pull requests welcome. The project is small on purpose —
nine properties grounded in social-choice theory, two extras for the
soft-voting case, no scope creep toward classifier-level audits.

## License

MIT.
