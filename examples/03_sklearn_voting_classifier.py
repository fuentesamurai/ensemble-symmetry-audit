"""Example 3: auditing a real sklearn VotingClassifier.

A common ML pattern: train three different classifiers and combine their
predictions with sklearn.ensemble.VotingClassifier. The aggregator looks
trivial — hard-voting majority — yet the per-property audit reveals
which structural assumptions hold and which do not.

This example does NOT inspect or modify the classifiers' training. It
audits only the aggregation step, by wrapping the trained VotingClassifier
in a thin function that takes a list of predicted labels and returns
the ensemble's choice. That is exactly the surface this library is
designed to test.

Requires scikit-learn:

    pip install -e ".[dev]"
"""

import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from ensemble_symmetry_audit import audit


def build_voting_classifier(seed: int = 0):
    """Train a hard-voting ensemble on a synthetic 3-class problem."""
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
    clf = VotingClassifier(estimators=estimators, voting="hard")
    clf.fit(X, y)
    return clf


def make_aggregator(clf):
    """Wrap a trained VotingClassifier as a vote-list -> decision function.

    The library tests aggregation, not classification. So instead of
    feeding the classifier feature vectors, we directly hand it lists of
    integer class labels (as if the underlying estimators had already
    voted) and reproduce sklearn's hard-voting rule:
    `argmax(bincount(predictions))`.
    """
    classes_ = clf.classes_

    def aggregate(votes):
        # `votes` is a list of integer labels, one per "estimator".
        # Reproduce sklearn's hard-voting tie-break: numpy argmax on
        # bincount, which is positional (first-class advantage).
        arr = np.asarray(votes, dtype=int)
        return int(classes_[np.argmax(np.bincount(arr, minlength=len(classes_)))])

    return aggregate


def main():
    clf = build_voting_classifier(seed=0)
    aggregate = make_aggregator(clf)
    classes = list(int(c) for c in clf.classes_)

    print(f"Auditing sklearn VotingClassifier hard-voting rule on classes {classes}")
    print(f"(3 underlying estimators: LogisticRegression, DecisionTree, GaussianNB)")
    print("-" * 60)

    report = audit(aggregate, classes=classes, n_voters=3, seed=42)
    print(report)
    print()
    print("Takeaways:")
    print("- sklearn's hard-voting tie-break is np.argmax(bincount(...)),")
    print("  which silently favours lower-indexed labels on ties.")
    print("- The detectors quantify exactly when and how often this")
    print("  matters for your class set and voter count.")
    print()
    print("To export as JSON for CI:")
    print("  report.to_json()  -> machine-readable audit log")


if __name__ == "__main__":
    main()
