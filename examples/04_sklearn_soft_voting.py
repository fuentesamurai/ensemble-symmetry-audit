"""Example 4: auditing sklearn VotingClassifier(voting='soft') in one line.

Same one-call adapter as example 03 — `audit_sklearn_classifier` picks
the soft-voting suite automatically when ``clf.voting == 'soft'``.

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
        voting="soft",
    ).fit(X, y)

    print("Auditing sklearn VotingClassifier(voting='soft') in one call")
    print("-" * 60)
    report = audit_sklearn_classifier(clf, seed=42)
    print(report)

    print()
    print("Soft voting (averaging predict_proba) passes the structural")
    print("audit because real-valued probability averaging rarely")
    print("produces exact ties for the argmax tie-break to bite.")


if __name__ == "__main__":
    main()
