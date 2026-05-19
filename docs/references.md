# References

Academic references for the properties tested by this library.

## Social choice theory

- **Kenneth J. Arrow**, *Social Choice and Individual Values* (Wiley,
  1951). Original source of the Independence of Irrelevant Alternatives
  (IIA) condition and Arrow's impossibility theorem: no deterministic
  social choice function over three or more alternatives can
  simultaneously satisfy unanimity, IIA, non-dictatorship, and
  universal domain.

- **Kenneth O. May**, "A Set of Independent Necessary and Sufficient
  Conditions for Simple Majority Decision", *Econometrica* 20 (4),
  680–684 (1952). Characterises majority rule via four axioms:
  anonymity, neutrality, positive responsiveness, and decisiveness.

- **William V. Gehrlein**, *Condorcet's Paradox* (Springer, 2006).
  Comprehensive treatment of voting paradoxes including the no-show
  paradox (participation monotonicity violation), Condorcet cycles,
  and the failure of IIA under common voting rules.

- **Hervé Moulin**, "Condorcet's Principle Implies the No Show
  Paradox", *Journal of Economic Theory* 45 (1), 53–64 (1988).
  Proves the no-show paradox is unavoidable in any Condorcet-consistent
  rule.

## Property-based testing

- **John Hughes and Koen Claessen**, "QuickCheck: A Lightweight Tool
  for Random Testing of Haskell Programs", *Proceedings of the Fifth
  ACM SIGPLAN International Conference on Functional Programming*,
  268–279 (2000). The paper that introduced property-based testing as
  a mainstream technique.

- **David R. MacIver et al.**, *Hypothesis* — Python implementation
  of property-based testing with automatic counterexample shrinking.
  [hypothesis.readthedocs.io](https://hypothesis.readthedocs.io/).

## Related Python tooling

- **`fairlearn`**, **`aif360`** — demographic-fairness audit libraries
  (different problem from structural-symmetry audit). If your concern
  is whether a classifier discriminates by race, gender, or other
  protected attributes, those are the right tools, not this one.

- **`hypothesis`** — already cited above as the underlying engine for
  the optional `shrink_*` helpers.

- **`scipy.stats`** — provides the `chisquare` and `binomtest`
  routines used by `balanced_input_symmetry` and
  `null_majority_abstention`.
