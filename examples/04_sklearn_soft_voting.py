"""Example 4: auditing sklearn VotingClassifier(voting='soft').

Soft voting averages the per-class probabilities produced by each
underlying estimator and picks the argmax. This is the rule scikit-learn
uses when the estimators all implement `predict_proba`.

We wrap the trained classifier into a vote-aggregator that takes a list
of probability dicts (one per estimator) and returns the chosen class.
The `soft_audit()` battery then exercises Pareto, balance, regime flip,
monotonicity, permutation invariance, and continuity on this aggregator.

Requires scikit-learn:

    pip install -e ".[dev]"
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from ensemble_symmetry_audit import soft_audit


def build_soft_voting_classifier(seed: int = 0):
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=6,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=seed,
    )
    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
        ("dt", DecisionTreeClassifier(random_state=seed)),
        ("nb", GaussianNB()),
    ]
    clf = VotingClassifier(estimators=estimators, voting="soft")
    clf.fit(X, y)
    return clf


def make_soft_aggregator(clf):
    """Wrap the trained VotingClassifier as a vote-list -> class function
    operating on probability dicts.

    `votes` is a list of {class: prob} dicts, one per "voter" (estimator).
    sklearn's soft-voting rule averages the per-class probabilities and
    returns the class whose averaged probability is highest.
    """
    classes_ = list(int(c) for c in clf.classes_)

    def aggregate(votes):
        summed = np.zeros(len(classes_))
        for v in votes:
            for i, c in enumerate(classes_):
                summed[i] += v.get(c, 0.0)
        idx = int(np.argmax(summed))
        return classes_[idx]

    return aggregate, classes_


def main():
    clf = build_soft_voting_classifier(seed=0)
    aggregate, classes = make_soft_aggregator(clf)

    flip = None
    if len(classes) == 2:
        flip = {classes[0]: classes[1], classes[1]: classes[0]}

    print(f"Auditing sklearn VotingClassifier(voting='soft') on classes {classes}")
    print("3 underlying estimators: LogisticRegression, DecisionTree, GaussianNB")
    print("-" * 60)

    report = soft_audit(
        aggregate,
        classes=classes,
        n_voters=len(clf.estimators_),
        flip_map=flip,
        seed=42,
    )
    print(report)
    print()
    print("Takeaways:")
    print("- Soft voting (averaging probability vectors then argmax) is")
    print("  the textbook well-behaved aggregator. It should pass every")
    print("  structural property unless your estimators produce")
    print("  pathological probability distributions.")
    print("- Compare with examples/03 (hard voting) to see the difference.")
    print()
    print("To export as JSON for CI:")
    print("  report.to_json()")


if __name__ == "__main__":
    main()
