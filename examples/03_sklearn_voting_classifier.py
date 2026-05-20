"""Example 3: auditing sklearn VotingClassifier(voting='hard') in one line.

v0.5+ ships first-class adapters, so the audit is now:

    from ensemble_symmetry_audit import audit_sklearn_classifier
    report = audit_sklearn_classifier(trained_clf, seed=42)

That single call detects the classifier kind (Voting hard / soft,
Bagging, RandomForest, ExtraTrees, generic predict_proba), picks the
matching adapter, picks `audit()` or `soft_audit()`, and returns the
full ``AuditReport``.

Requires:

    pip install ensemble-symmetry-audit[sklearn]
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from ensemble_symmetry_audit import audit_sklearn_classifier


def main():
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=6,
        n_classes=3, n_clusters_per_class=1, random_state=0,
    )
    clf = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=0)),
            ("dt", DecisionTreeClassifier(random_state=0)),
            ("nb", GaussianNB()),
        ],
        voting="hard",
    ).fit(X, y)

    print("Auditing sklearn VotingClassifier(voting='hard') in one call")
    print("Underlying estimators: LogisticRegression, DecisionTree, GaussianNB")
    print("-" * 60)

    report = audit_sklearn_classifier(clf, seed=42)
    print(report)

    print()
    print("As expected, the positional argmax tie-break produces a")
    print("structural advantage for class 0. The full grid is in")
    print("examples/case_study_sklearn.py.")


if __name__ == "__main__":
    main()
