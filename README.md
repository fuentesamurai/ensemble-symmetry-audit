# ensemble-symmetry-audit

[![tests](https://github.com/fuentesamurai/ensemble-symmetry-audit/actions/workflows/tests.yml/badge.svg)](https://github.com/fuentesamurai/ensemble-symmetry-audit/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Property-based audit of voting ensemble aggregators.

A small Python library that audits any voting, majority, or
weighted-vote aggregator against **nine structural properties** (four
classical axioms + three distributional tests + two implementation
invariants = 9) and reports a minimal counterexample whenever one
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
         statistic: chi2=369.8, p_value=0.0, alpha=0.01,
                    cohens_w=0.430, max_relative_deviation=0.430
         counterexample: {'observed_counts': {'UP': 570, 'DOWN': 1430}}
[FAIL] regime_flip_invariance      (500 cases)
         counterexample: {'out_original': 'DOWN', 'out_flipped': 'DOWN',
                          'expected_flipped': 'UP'}
```

The report tells you which property failed, the test statistic and
**effect size** that detected it, and a minimal concrete input that
reproduces the failure.

## Where does my aggregator stand?

Run the same audit against five canonical aggregators on K=3 classes,
N=5 voters (reproducible from `examples/comparison_table.py`):

| Aggregator                                              | Result | Fails                          |
|---------------------------------------------------------|--------|--------------------------------|
| Plurality + multiset-hash tie-break                     | 10/11  | IIA (Arrow's theorem)          |
| Plurality + alphabetical tie-break                      | 9/11   | balance, IIA                   |
| `Counter.most_common()` (insertion-order tie-break)     | 9/11   | permutation, IIA               |
| sklearn `VotingClassifier(voting='hard')`               | 9/11   | balance, IIA                   |
| Constant aggregator (always returns first class)        | 9/11   | Pareto, balance                |

Five aggregators, five different failure patterns, none pass
everything. That is Arrow's theorem in practice — the library tells
you *which* trade-off each rule made.

(The count is 11 not 9 because `monotonicity` and
`participation_monotonicity` expand to one result per class in the
3-class configuration. The library has 9 named property families.)

## Install

```
pip install ensemble-symmetry-audit
```

Python 3.10+. Requires `scipy>=1.10` and `numpy>=1.20`.

For optional Hypothesis-driven counterexample shrinking:

```
pip install ensemble-symmetry-audit[shrink]
```

## Audit a trained sklearn classifier in one line

v0.5 ships first-class adapters. Skip the wrapper boilerplate:

```python
from sklearn.ensemble import VotingClassifier
from ensemble_symmetry_audit import audit_sklearn_classifier

clf = VotingClassifier(estimators=[...], voting="hard").fit(X, y)

report = audit_sklearn_classifier(clf, seed=42)
print(report)
```

`audit_sklearn_classifier` detects the classifier type (Voting,
Bagging, RandomForest, ExtraTrees, or anything else exposing
`predict_proba` + `classes_`), picks the matching adapter, picks
`audit()` or `soft_audit()`, and runs the full battery in one call.

Specialised adapters are also exported individually in
`ensemble_symmetry_audit.adapters.sklearn` if you want to wire the
audit yourself. XGBoost and LightGBM have parallel adapters in
`adapters.xgboost` and `adapters.lightgbm`.

```
pip install ensemble-symmetry-audit[sklearn]   # sklearn adapters
pip install ensemble-symmetry-audit[xgboost]   # XGBoost adapter
pip install ensemble-symmetry-audit[lightgbm]  # LightGBM adapter
pip install ensemble-symmetry-audit[adapters]  # all of the above
```

## Quick start (custom aggregator)

A clean audit on a well-behaved aggregator (binary majority, odd voter
count — no ties possible):

```python
from collections import Counter
from ensemble_symmetry_audit import audit


def fair_majority(votes):
    return Counter(votes).most_common(1)[0][0]


report = audit(
    fair_majority,
    classes=["UP", "DOWN"],
    n_voters=11,
    flip_map={"UP": "DOWN", "DOWN": "UP"},
)

print(report)
```

Output:

```
Ensemble symmetry audit report
--------------------------------------------------
[PASS] pareto_unanimity            (2 cases)
[PASS] balanced_input_symmetry     (2000 cases)
         statistic: chi-squared, p=0.79, cohens_w=0.01
[PASS] regime_flip_invariance      (500 cases)
[PASS] monotonicity[UP]            (200 cases)
[PASS] monotonicity[DOWN]          (200 cases)
[PASS] participation_monotonicity[UP]    (~100 eligible cases)
[PASS] participation_monotonicity[DOWN]  (~100 eligible cases)
[PASS] permutation_invariance      (200 cases, mode=transpositions)
[PASS] tie_break_determinism       (200 cases)
--------------------------------------------------
ALL PROPERTIES HELD
```

For CI integration:

```python
report.all_passed        # bool
report.failed            # list of failing DetectorResults
report.to_json()         # JSON for machine consumption
```

## The sklearn finding

Pointing the audit at `sklearn.ensemble.VotingClassifier(voting='hard')`
produced a quantifiable, reproducible result that is worth flagging up
front (see `examples/case_study_sklearn.py`):

- Under uniform random inputs, with K=10 classes and N=3 voters, class
  0 wins **23.8% of decisions** instead of the expected 10% — a +138%
  relative advantage from the positional `np.argmax` tie-break.
- All twelve `(K, N)` configurations tested fail balanced-input
  symmetry; soft voting passes in eleven of twelve.

To be clear: sklearn's `np.argmax(np.bincount(...))` is a documented,
deterministic design choice, not a bug. The library quantifies the
**structural cost of that choice** under symmetric input — useful when
you need to know whether the positional tie-break matters for your
class set and voter count, and worth knowing before shipping a hard
voting ensemble into a domain where it would.

## What it tests

Nine named properties, grouped by origin and purpose:
**4 classical axioms + 3 distributional tests + 2 implementation invariants = 9.**

### Classical axioms (social-choice theory)

| Property                                          | Reference                | What it catches                                                                  |
|---------------------------------------------------|--------------------------|----------------------------------------------------------------------------------|
| **Pareto / unanimity**                            | May 1952                 | Aggregators that override a unanimous vote                                       |
| **Monotonicity**                                  | May 1952                 | Flipping a vote toward X moving the winner away from X                           |
| **Participation monotonicity**                    | Moulin 1988              | No-show paradox: adding an X-voter making X lose                                 |
| **Independence of Irrelevant Alternatives (IIA)** | Arrow 1951               | Adding or removing a losing class flipping the winner                            |

### Distributional / statistical tests

| Property                                              | Test            | What it catches                                                            |
|-------------------------------------------------------|-----------------|----------------------------------------------------------------------------|
| **Balanced-input symmetry**                           | chi-squared     | A silent skew toward one class under uniform random input                  |
| **Null-majority abstention** *(opt-in)*               | binomial        | Picking sides when the evidence is balanced (where abstention is permitted) |
| **Regime-flip invariance**                            | counterexample  | Asymmetric reaction to user-defined label permutations (see note below)    |

### Implementation invariants

| Property                                                  | What it catches                                            |
|-----------------------------------------------------------|------------------------------------------------------------|
| **Permutation invariance** *(adjacent-transpositions)*    | Output depending on voter order — exhaustive via S_n generators |
| **Tie-break determinism**                                 | Same input producing different outputs across runs        |

The chi-squared and binomial tests report **effect size** (Cohen's w,
max relative deviation) alongside the p-value. At `n_trials = 2000`,
chi-squared rejects vanishingly small biases that may not be
practically actionable; reading p-value and effect size together
distinguishes "statistically significant but tiny" from "structurally
important".

#### Note on regime-flip invariance

The user supplies a `flip_map` declaring which label permutation the
aggregator is expected to respect. Common cases:

- `{"BUY": "SELL", "SELL": "BUY", "HOLD": "HOLD"}` — directional flip
  for ternary trading decisions where HOLD is neutral-fixed.
- `{"UP": "DOWN", "DOWN": "UP"}` — binary symmetric.
- `{"A": "B", "B": "A"}` for any pair where you've decided the labels
  are interchangeable from the aggregator's point of view.

Not every aggregator has a natural flip. For `{spam, ham}` there is
no symmetry to test — flipping "spam" to "ham" is meaningful business
asymmetry, not noise — and you simply omit `flip_map`. The property
is opt-in and exists for cases where the aggregation rule is supposed
to be invariant under a label-level symmetry of the application domain.

## Reproducibility

Every detector accepts a `seed` parameter (default `42`) that
deterministically seeds the random number generator. The default is
non-`None` on purpose: CI builds that re-run the audit must produce
byte-identical reports, otherwise a "flaky" property test will be
randomly dismissed as noise. Override with your own seed when you want
to vary the search, set `seed=None` is not supported.

`audit()` and `soft_audit()` expose a single base seed and derive
per-detector seeds from it.

### Choosing `n_trials` (statistical power)

The default `n_trials = 2000` for `balanced_input_symmetry` has roughly
80% power to detect skews of ~10% relative deviation. Smaller skews
need more trials. The library ships a helper that returns the
minimum `n_trials` for a given target skew:

```python
from ensemble_symmetry_audit import min_n_trials_for_balance

# How many trials to reliably catch a 51/49 binary skew (deviation = 0.02)?
min_n_trials_for_balance(n_classes=2, max_relative_deviation=0.02)
# ~29,000

# 5% deviation in 3-class:
min_n_trials_for_balance(n_classes=3, max_relative_deviation=0.05)
# ~11,000

# 10% deviation in 10-class:
min_n_trials_for_balance(n_classes=10, max_relative_deviation=0.10)
# ~19,000
```

| Skew you want to catch       | K=2     | K=3     | K=10    |
|------------------------------|---------|---------|---------|
| ±10% (e.g. 55/45 in binary)  | ~1,200  | ~2,800  | ~19,000 |
| ±5%                          | ~4,600  | ~11,000 | ~77,000 |
| ±2% (e.g. 51/49 in binary)   | ~29,000 | ~69,000 | ~480,000|

(Default α=0.01, power=0.8. The helper uses non-central chi-squared
to compute the minimum n; rounding is upward.)

If your application cares about tiny skews — quant finance with K=2,
risk decisions — bump `n_trials` accordingly. The default is
deliberately calibrated to catch the "structurally important" range,
not the "statistically significant but tiny" range.

### Computational cost

Most detectors scale linearly in `n_voters` and `n_trials`. The one
worth knowing about:

- `permutation_invariance(mode="transpositions")` runs
  `(n_voters - 1) * n_trials` calls to the aggregator. With the default
  `n_trials = 200`, that means ~2,000 calls at `n_voters = 11` and
  ~4,000 at `n_voters = 21`. Fine for any aggregator that runs in
  microseconds; consider lowering `n_trials` if your aggregator is
  expensive (e.g., calls an external service per evaluation).

## Soft voting

The same properties (plus two soft-only ones — `soft_continuity` and
`soft_participation_monotonicity`) are available for soft /
probabilistic aggregators via `soft_audit()`. See
[docs/soft-voting.md](docs/soft-voting.md).

## Three or more classes: a note on Arrow

Arrow's impossibility theorem (1951) is stated for social welfare
functions over **ordinal preferences** with a universal domain. The
aggregators tested here typically receive **categorical votes**
without an underlying preference ranking, so the literal Arrow theorem
does not transfer pointwise. However, the *intuition* — that you
cannot simultaneously satisfy unanimity, IIA, anonymity, and a
non-trivial decision rule over three or more alternatives — survives
in the categorical setting and is exactly what this library
quantifies. `examples/02_three_class_classifier.py` shows three
innocent-looking three-class aggregators failing different properties
each: no deterministic three-class rule passes the full suite, and
the library tells you which trade-off each rule made.

## Examples

| File                                       | What it shows                                                |
|--------------------------------------------|--------------------------------------------------------------|
| `examples/01_catching_a_phantom_voter.py`  | Audit catching a hidden directional bias                     |
| `examples/02_three_class_classifier.py`    | Three-class aggregators with three different failure modes   |
| `examples/03_sklearn_voting_classifier.py` | Audit on sklearn `VotingClassifier(voting='hard')` (K=3)     |
| `examples/04_sklearn_soft_voting.py`       | Audit on sklearn `VotingClassifier(voting='soft')`           |
| `examples/case_study_sklearn.py`           | Grid audit across (K, N) of sklearn hard + soft voting       |
| `examples/comparison_table.py`             | Generates the at-a-glance comparison table above             |

Examples 03 and 04 use the v0.5 one-call adapter (`audit_sklearn_classifier`).
The case study and comparison table use the lower-level audit
functions directly to show how to build custom comparisons.

## Documentation

- [docs/soft-voting.md](docs/soft-voting.md) — soft-voting suite,
  `soft_continuity` and the boundary problem
- [docs/hypothesis-shrinking.md](docs/hypothesis-shrinking.md) —
  optional Hypothesis-driven counterexample shrinking and strategies
- [docs/references.md](docs/references.md) — Arrow, May, Moulin,
  Gehrlein, QuickCheck, and related tooling

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

**Latest — v0.5.0** (May 2026):
- First-class adapters for sklearn `VotingClassifier`,
  `BaggingClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`,
  plus a generic `adapt_argmax_proba_classifier` and adapters for
  XGBoost / LightGBM.
- One-call convenience `audit_sklearn_classifier(clf, seed=42)` that
  detects the classifier type and runs the right audit suite.
- Examples 03 / 04 now use the adapters — audit is a single line.
- New optional extras: `[sklearn]`, `[xgboost]`, `[lightgbm]`,
  `[adapters]` (all three).

**Previous — v0.4.3:**
- GitHub Actions CI running pytest on Python 3.10 / 3.11 / 3.12 and
  exercising every example script on every push and PR (badge above).
- `min_n_trials_for_balance(K, deviation, alpha, power)` helper +
  guidance table for choosing `n_trials` from a target skew.
- IIA result on binary input now spells out *why* it is skipped.

**Previous — v0.4.2:**
- At-a-glance comparison table in README + reproducible
  `examples/comparison_table.py`.
- Clean-pass Quick Start example, examples index, computational-cost
  notes.

**Previous — v0.4.1:**
- Effect-size reporting (Cohen's w, max relative deviation) alongside
  p-values.
- Property grouping (axioms / tests / invariants), Arrow nuance,
  reproducibility guarantees.

**Previous — v0.4.0:**
- `participation_monotonicity` detector (hard + soft).
- `soft_continuity` boundary-aware.
- `permutation_invariance` exhaustive transpositions.
- Hypothesis as optional `[shrink]` extra.

**Next:**
- **v0.6** — CI reporters (JUnit XML, GitHub Actions annotations) and
  HTML / Markdown report exporters.
- **v0.7** — soft-vote calibration property tests.
- **v0.8** — `StackingClassifier` adapter (meta-learner semantics
  require their own audit shape).

**v1.0 milestone** — API stability commitment and adapter coverage
for the four most-used sklearn / XGBoost / LightGBM ensemble classes.
After v1.0, breaking changes follow semver and require a major bump.

See [Releases](https://github.com/fuentesamurai/ensemble-symmetry-audit/releases)
for the full version history.

## Contributing

Issues and pull requests welcome. The project is small on purpose —
nine properties grounded in social-choice theory and software-property
testing, plus two extras for the soft-voting case.

## License

MIT.
