# Soft voting

If your aggregator operates on probability distributions rather than
categorical labels — e.g. averaging `predict_proba` outputs and taking
the argmax, as `sklearn.ensemble.VotingClassifier(voting="soft")` does —
use `soft_audit()`:

```python
from ensemble_symmetry_audit import soft_audit


def my_soft_aggregator(votes):
    # votes: List[Dict[class, prob]]
    summed = {}
    for v in votes:
        for c, p in v.items():
            summed[c] = summed.get(c, 0.0) + p
    return max(summed, key=summed.get)


report = soft_audit(my_soft_aggregator,
                    classes=["A", "B", "C"],
                    n_voters=5,
                    seed=42)
print(report)
```

The soft suite mirrors the hard suite with two soft-only properties:

| #  | Property                                       | What it catches                                                                  |
|----|------------------------------------------------|----------------------------------------------------------------------------------|
| 1  | `soft_pareto_unanimity`                        | Aggregator that ignores high-confidence unanimous voters                         |
| 2  | `soft_balanced_input_symmetry`                 | Skew under Dirichlet-uniform random probabilities (chi-squared test)            |
| 3  | `soft_regime_flip_invariance`                  | Asymmetry under probability permutation                                          |
| 4  | `soft_monotonicity`                            | Moving probability mass toward X moving output away from X                       |
| 5  | `soft_participation_monotonicity` *(new v0.4)* | Adding a high-confidence X voter moving the winner away from X (no-show paradox) |
| 6  | `soft_permutation_invariance`                  | Output depending on voter order                                                  |
| 7  | `soft_continuity` *(new v0.3; refined v0.4)*   | Small probability perturbations flipping the output *far from the boundary*      |

## `soft_continuity` and the boundary problem

A naive continuity test flags every perturbation that flips the
output. That generates false positives: when a ballot is *already* at
a tight tie between two classes, an `ε = 1e-3` perturbation can flip
the winner and there is no bug — the perturbation simply crossed the
decision boundary.

`soft_continuity` computes the **decision margin** (the gap between
winner and runner-up averaged probabilities) for every random ballot
and partitions cases into two buckets:

- `robust_cases`: margin > `margin_threshold`. Here ε-perturbations
  should not flip the output. Flip rate must stay under `tolerance`
  (default 1%) for the property to pass.
- `near_boundary_cases`: margin ≤ `margin_threshold`. Flips here are
  mathematically expected and counted separately for information.

Tune `margin_threshold` and `epsilon` to your application — the
defaults (`0.02` and `1e-3`) are reasonable for K=3 and small N, but
real-world calibrated classifiers may need different values.

## Example: auditing sklearn

`examples/04_sklearn_soft_voting.py` audits
`VotingClassifier(voting="soft")` and demonstrates the contrast with
the hard-voting case: soft voting passes every structural property
because real-valued probability averaging rarely produces exact ties
for the positional tie-break to bite.

`examples/case_study_sklearn.py` quantifies this empirically across a
grid of (n_classes, n_voters): hard voting fails balance in 12/12
configurations, soft voting in only 1/12.
