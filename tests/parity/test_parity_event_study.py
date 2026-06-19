"""Parity tests against independent reference implementations.

These pin the hand-rolled kernels to trusted references (the brief's parity
plan):

- market-model abnormal returns vs. a ``statsmodels`` OLS reference;
- the HAC standard error vs. the reused ``fedcausal.evaluation.hac`` to 1e-10;
- the Boehmer-Musumeci-Poulsen statistic vs. a hand reference;
- Benjamini-Hochberg / Romano-Wolf vs. a reference.

Authored sequentially once the compute kernels land — skipped in the scaffold.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.parity

_SCAFFOLD_REASON = "scaffold: parity body authored once compute kernels land"


def test_market_model_abnormal_returns_vs_statsmodels_ols() -> None:
    """Market-model abnormal returns match a statsmodels OLS reference."""
    pytest.skip(_SCAFFOLD_REASON)


def test_hac_se_matches_reference_to_1e_10() -> None:
    """The CAR HAC SE matches the reused ``newey_west_se`` reference to 1e-10."""
    pytest.skip(_SCAFFOLD_REASON)


def test_bmp_statistic_vs_hand_reference() -> None:
    """The BMP standardized-residual statistic matches a hand-computed reference."""
    pytest.skip(_SCAFFOLD_REASON)


def test_benjamini_hochberg_and_romano_wolf_vs_reference() -> None:
    """BH / Romano-Wolf adjusted p-values match a reference implementation."""
    pytest.skip(_SCAFFOLD_REASON)
