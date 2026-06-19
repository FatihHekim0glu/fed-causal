"""Parity for the multiple-testing corrections (the brief's parity plan).

- Benjamini-Hochberg adjusted p-values and rejections match
  ``statsmodels.stats.multitest.multipletests(method="fdr_bh")`` exactly.
- Romano-Wolf reproduces a hand-rolled stepdown reference and, critically,
  CONTROLS the family-wise error rate on a global-null grid (the FWER property
  the verdict relies on) while still rejecting a clearly-significant spec.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from fedcausal._rng import make_rng
from fedcausal.evaluation.multiple_testing import benjamini_hochberg, romano_wolf

pytestmark = pytest.mark.parity


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg vs statsmodels                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pvalues",
    [
        [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216],
        [0.9, 0.8, 0.7, 0.6, 0.5],  # nothing significant
        [0.0001, 0.0002, 0.0003],  # everything significant
        [0.04, 0.04, 0.04, 0.04],  # ties
        [0.5],  # singleton
    ],
)
def test_benjamini_hochberg_matches_statsmodels(pvalues: list[float]) -> None:
    """BH adjusted p-values and the rejection mask match statsmodels exactly."""
    arr = np.asarray(pvalues, dtype=float)
    result = benjamini_hochberg(arr, alpha=0.05)

    rej_ref, padj_ref, _, _ = multipletests(arr, alpha=0.05, method="fdr_bh")

    np.testing.assert_allclose(result.adjusted_pvalues, padj_ref, atol=1e-12, rtol=0.0)
    assert np.array_equal(result.rejected, rej_ref)
    assert result.any_survives == bool(rej_ref.any())
    assert result.n_tests == arr.size  # honest grid size


def test_benjamini_hochberg_adjusted_are_monotone_in_raw_order() -> None:
    """Step-up enforcement: adjusted p-values are non-decreasing in raw rank."""
    arr = np.array([0.01, 0.02, 0.03, 0.20, 0.04], dtype=float)
    result = benjamini_hochberg(arr, alpha=0.1)
    sorted_adj = result.adjusted_pvalues[np.argsort(arr, kind="stable")]
    assert np.all(np.diff(sorted_adj) >= -1e-12)


# --------------------------------------------------------------------------- #
# Romano-Wolf stepdown                                                         #
# --------------------------------------------------------------------------- #
def _romano_wolf_reference(
    statistics: np.ndarray,
    null_distributions: np.ndarray,
) -> np.ndarray:
    """Independent stepdown reference (two-sided, mean-centred maxT)."""
    abs_stats = np.abs(statistics)
    centred = np.abs(null_distributions - null_distributions.mean(axis=1, keepdims=True))
    order = np.argsort(abs_stats)[::-1]
    adj = np.empty(statistics.size, dtype=float)
    running = 0.0
    for step, idx in enumerate(order):
        active = order[step:]
        max_null = centred[active].max(axis=0)
        tail = float(np.mean(max_null >= abs_stats[idx]))
        running = max(running, tail)
        adj[idx] = min(running, 1.0)
    return adj


def test_romano_wolf_matches_hand_reference() -> None:
    """Romano-Wolf adjusted p-values match an independent stepdown reference."""
    rng = make_rng(11)
    null = rng.standard_normal((5, 3000))
    statistics = np.array([3.5, 0.2, -2.9, 0.1, 1.0], dtype=float)
    result = romano_wolf(statistics, null, alpha=0.05)
    reference = _romano_wolf_reference(statistics, null)
    np.testing.assert_allclose(result.adjusted_pvalues, reference, atol=1e-12, rtol=0.0)


def test_romano_wolf_controls_fwer_on_global_null() -> None:
    """FWER control: on a global-null grid, false rejections are rare (<= alpha).

    Across many independent global-null grids the empirical probability of ANY
    rejection must not exceed the nominal family-wise error rate (with Monte-Carlo
    slack). This is the property the honest-null verdict leans on.
    """
    alpha = 0.05
    n_specs = 8
    n_resamples = 400
    n_trials = 300
    rng = make_rng(2026)

    false_rejections = 0
    for _ in range(n_trials):
        null = rng.standard_normal((n_specs, n_resamples))
        # Observed statistics ARE draws from the same null (global null is true).
        statistics = rng.standard_normal(n_specs)
        result = romano_wolf(statistics, null, alpha=alpha)
        if result.any_survives:
            false_rejections += 1

    fwer = false_rejections / n_trials
    # Allow Monte-Carlo slack above the nominal level.
    assert fwer <= alpha + 0.04


def test_romano_wolf_rejects_a_clearly_significant_spec() -> None:
    """A spec far in the joint-null tail is rejected (power, not just control)."""
    rng = make_rng(7)
    null = rng.standard_normal((6, 4000))
    statistics = null.mean(axis=1).copy()
    statistics[3] = 8.0  # unambiguously extreme
    result = romano_wolf(statistics, null, alpha=0.05)
    assert result.rejected[3]
    assert result.any_survives
    assert result.n_tests == 6
