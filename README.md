# ensemble-symmetry-audit

Property-based audit of voting ensemble aggregators.

A small Python library that audits any voting, majority, or
weighted-vote aggregator against nine structural properties and
reports a minimal counterexample whenever one fails. Hard *and* soft
(probabilistic) voting are both supported.

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

## The sklearn finding

The library was first written to audit private code. Pointing it at
`sklearn.ensemble.VotingClassifier(voting='hard')` produced a
quantifiable, reproducible result that is worth flagging up front
(see `examples/case_study_sklearn.py`):

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

Nine properties, grouped by origin and purpose.

### Classical axioms (social-choice theory)

| Property                                          | Reference                | What it catches                                                                  |
|---------------------------------------------------|--------------------------|----------------------------------------------------------------------------------|
| **Pareto / unanimity**                            | May 1952                 | Aggregators that override a unanimous vote                                       |
| **Monotonicity**                                  | May 1952                 | Flipping a vote toward X moving the winner away from X                           |
| **Participation monotonicity** *(new v0.4)*       | Moulin 1988              | No-show paradox: adding an X-voter making X lose                                 |
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
asymmetry, not noise — and you simply omit `flip_map`. The property is
opt-in and exists for cases where the aggregation rule is supposed to
be invariant under a label-level symmetry of the application domain.

## Reproducibility

Every detector accepts a `seed` parameter (default `42`) that
deterministically seeds the random number generator used to draw
inputs. Given the same seed, repeated runs of the same configuration
produce identical `DetectorResult` outputs. `audit()` and `soft_audit()`
expose a single base seed and derive per-detector seeds from it, so
two CI runs with the same base seed produce byte-identical reports.

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

**Latest — v0.4.1** (May 2026):
- Effect-size reporting (Cohen's w, max relative deviation) alongside
  p-values in the chi-squared tests.
- Properties grouped by category in README. Regime-flip semantics and
  Arrow's applicability clarified.
- Reproducibility guarantees documented.

**Previous — v0.4.0:**
- `participation_monotonicity` detector (hard + soft).
- `soft_continuity` boundary-aware (no more false positives near tight ties).
- `permutation_invariance` defaults to exhaustive adjacent transpositions.
- Hypothesis moved to the optional `[shrink]` extra.

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
nine properties grounded in social-choice theory and software-property
testing, plus two extras for the soft-voting case.

## License

MIT.
