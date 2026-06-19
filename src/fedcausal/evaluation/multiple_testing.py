"""Multiple-testing corrections across the FULL specification grid.

The event study is run across a grid of specifications — event windows x expected-
return models x surprise subsets — so the family-wise / false-discovery error must
be controlled honestly across the WHOLE grid, not cherry-picked from the single
most significant cell.

Two corrections are provided:

- **Benjamini-Hochberg (1995)** step-up FDR control; and
- **Romano-Wolf (2005)** stepdown, which controls the family-wise error rate
  using a bootstrap of the joint null and is more powerful than Bonferroni under
  dependence (the spec cells are correlated).

HONEST ``n_tests``: the correction counts EVERY cell actually evaluated. A
spec is only declared significant if it survives the correction across the full
grid; this is an input to the PURE verdict.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from fedcausal._constants import DEFAULT_ALPHA

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    """The outcome of a multiple-testing correction over the spec grid.

    Attributes
    ----------
    method:
        ``"benjamini_hochberg"`` or ``"romano_wolf"``.
    n_tests:
        The honest total number of specifications tested (the full grid size).
    adjusted_pvalues:
        The corrected p-values, one per spec, in input order.
    rejected:
        Boolean mask of specs declared significant after correction.
    any_survives:
        Whether ANY spec survives the correction (an input to the verdict).
    alpha:
        The significance level the correction targeted.
    """

    method: str
    n_tests: int
    adjusted_pvalues: np.ndarray
    rejected: np.ndarray
    any_survives: bool
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` (lists, not arrays)."""
        payload = asdict(self)
        payload["adjusted_pvalues"] = [float(x) for x in self.adjusted_pvalues]
        payload["rejected"] = [bool(x) for x in self.rejected]
        payload["any_survives"] = bool(self.any_survives)
        payload["n_tests"] = int(self.n_tests)
        payload["alpha"] = float(self.alpha)
        return payload


def benjamini_hochberg(
    pvalues: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> MultipleTestingResult:
    """Benjamini-Hochberg step-up FDR correction over the spec grid.

    Parameters
    ----------
    pvalues:
        Raw per-spec p-values (one per grid cell).
    alpha:
        Target false-discovery rate.

    Returns
    -------
    MultipleTestingResult
        Adjusted p-values, the rejection mask, and ``any_survives`` with an honest
        ``n_tests`` equal to ``len(pvalues)``.

    Raises
    ------
    ValidationError
        If ``pvalues`` is empty or contains values outside ``[0, 1]``.
    """
    raise NotImplementedError


def romano_wolf(
    statistics: np.ndarray,
    null_distributions: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> MultipleTestingResult:
    """Romano-Wolf (2005) stepdown family-wise-error correction (bootstrap null).

    Uses a bootstrap of the joint null (e.g. the per-spec placebo distributions)
    to step down through the ordered statistics, controlling the family-wise error
    rate under dependence between the (correlated) spec cells.

    Parameters
    ----------
    statistics:
        The observed per-spec test statistics (one per grid cell).
    null_distributions:
        A ``(n_specs, n_resamples)`` array of joint-null draws for each spec
        (e.g. the placebo CARs per spec), used to derive the stepdown thresholds.
    alpha:
        Target family-wise error rate.

    Returns
    -------
    MultipleTestingResult
        Adjusted p-values, the rejection mask, and ``any_survives`` with an honest
        ``n_tests`` equal to ``len(statistics)``.

    Raises
    ------
    ValidationError
        If the inputs are misshaped or empty.
    """
    raise NotImplementedError
