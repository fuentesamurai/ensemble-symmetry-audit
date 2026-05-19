# Hypothesis-driven counterexample shrinking

Random sampling (the default mode of every detector) is fast and
sufficient for routine audits. But when a property fails you often
want the **smallest** input that breaks it, not the first one a
sampler happens to hit. v0.3 added two helpers that wrap
[`hypothesis.find()`](https://hypothesis.readthedocs.io/) and produce
adversarially-searched, automatically-shrunk counterexamples.

## Installation

Hypothesis is an **optional** runtime dependency. The core audit
suites run on numpy and scipy alone. Install the shrinking helpers
with:

```
pip install ensemble-symmetry-audit[shrink]
```

If you import `shrink_hard_counterexample` or `shrink_soft_counterexample`
without the extra installed, you get a clear `ImportError` pointing at
the same command above.

## Usage

```python
from ensemble_symmetry_audit import shrink_hard_counterexample


minimal = shrink_hard_counterexample(
    lambda votes: my_aggregator(votes) != "expected",
    classes=["A", "B", "C"],
    n_voters=11,
)
print(minimal)  # the smallest violating example, after shrinking
```

The first argument is a predicate that returns `True` when the input
violates the property under audit. Hypothesis searches for an input
satisfying the predicate, then automatically shrinks it to a minimal
example.

For soft (probabilistic) inputs:

```python
from ensemble_symmetry_audit import shrink_soft_counterexample


minimal = shrink_soft_counterexample(
    lambda votes: my_soft_aggregator(votes) != "expected",
    classes=["A", "B", "C"],
    n_voters=5,
)
```

## Strategies for your own tests

The strategies module is re-exported under
`ensemble_symmetry_audit.strategies` so you can drive your own
`@given`-decorated pytest tests directly:

```python
from hypothesis import given, settings
from ensemble_symmetry_audit.strategies import vote_lists


@given(vote_lists(classes=["A", "B", "C"], n_voters=11))
@settings(max_examples=500)
def test_my_aggregator_never_returns_None(votes):
    assert my_aggregator(votes) is not None
```

Available strategies:

- `vote_lists(classes, n_voters)`
- `balanced_vote_lists(classes, n_voters)` — every class appears at
  least once
- `probability_distributions(classes)` — single voter
- `probability_vote_lists(classes, n_voters)` — list of voters
- `concentrated_probability_vote_lists(classes, n_voters, target_class, confidence)` —
  every voter puts `confidence` mass on `target_class`

## When NOT to use shrinking

For routine audits across many configurations, random sampling is
faster (no search overhead) and the audit result is still informative
— you get a counterexample, just not the minimal one. Reach for
`shrink_*` when:

- A property failed and you want a publishable / debuggable example.
- You're writing a regression test against a known bug.
- You're driving `@given` tests yourself and want the shrinker.
