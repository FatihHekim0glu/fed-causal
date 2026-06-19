"""Parity tests for the event-study core (group ``eventstudy``).

Pins the hand-rolled kernels to trusted references:

- market-model ``alpha``/``beta``/``sigma`` vs. a ``statsmodels`` OLS reference to
  1e-8;
- abnormal returns vs. the closed-form ``actual - (alpha + beta * market)``;
- the HAC mean-CAR SE vs. the reused ``newey_west_se`` to 1e-10;
- the BMP statistic vs. a hand-computed standardized cross-sectional t.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from fedcausal.data.synthetic import SyntheticPanel
from fedcausal.evaluation.hac import newey_west_se
from fedcausal.events.windows import build_windows
from fedcausal.eventstudy.abnormal import abnormal_returns, fit_market_model
from fedcausal.eventstudy.tests import bmp_statistic, hac_car_test

pytestmark = pytest.mark.parity


@pytest.mark.parity
def test_market_model_abnormal_returns_vs_statsmodels_ols(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """Market-model parameters + abnormal returns match a statsmodels OLS reference."""
    panel = synthetic_event_panel
    grid = cast("pd.DatetimeIndex", panel.returns.index)
    w = build_windows(
        grid, panel.announcement_dates[5], event_half_width=2, estimation_window=120, estimation_gap=10
    )
    fitted = fit_market_model(panel.returns, panel.market, w, model="market")

    est_returns = panel.returns.iloc[w.estimation_start : w.estimation_end + 1]
    est_market = panel.market.iloc[w.estimation_start : w.estimation_end + 1]
    design = sm.add_constant(est_market.to_numpy())

    ev_returns = panel.returns.iloc[w.event_start : w.event_end + 1]
    ev_market = panel.market.iloc[w.event_start : w.event_end + 1]
    ar = abnormal_returns(panel.returns, panel.market, fitted, w)

    for col in panel.returns.columns:
        ols = sm.OLS(est_returns[col].to_numpy(), design).fit()
        assert fitted.alpha[col] == pytest.approx(ols.params[0], abs=1e-8)
        assert fitted.beta[col] == pytest.approx(ols.params[1], abs=1e-8)
        ref_sigma = float(np.sqrt(ols.ssr / (ols.nobs - 2)))
        assert fitted.sigma[col] == pytest.approx(ref_sigma, abs=1e-8)
        # abnormal returns equal the closed-form residual against the OLS line.
        ref_ar = ev_returns[col].to_numpy() - (ols.params[0] + ols.params[1] * ev_market.to_numpy())
        assert np.allclose(ar[col].to_numpy(), ref_ar, atol=1e-10)


@pytest.mark.parity
def test_hac_se_matches_reference_to_1e_10() -> None:
    """The CAR HAC SE matches the reused ``newey_west_se`` reference to 1e-10."""
    rng = np.random.default_rng(11)
    cars = rng.normal(0.004, 0.003, size=64)
    _mean, hac_se, _p = hac_car_test(cars)
    ref_se = newey_west_se(cars)
    assert hac_se == pytest.approx(ref_se, abs=1e-10)


@pytest.mark.parity
def test_hac_pvalue_matches_normal_tail() -> None:
    """The HAC p-value is the standard-normal two-sided tail of mean / HAC SE."""
    rng = np.random.default_rng(12)
    cars = rng.normal(0.006, 0.002, size=80)
    mean, hac_se, p_value = hac_car_test(cars)
    ref_p = float(2.0 * stats.norm.sf(abs(mean / hac_se)))
    assert p_value == pytest.approx(ref_p, abs=1e-12)


@pytest.mark.parity
def test_bmp_statistic_vs_hand_reference() -> None:
    """The BMP standardized-residual statistic matches a hand-computed reference."""
    # One event, three names, a two-day event window.
    abnormal = pd.DataFrame(
        [[0.010, -0.020, 0.030], [0.000, 0.010, -0.010]], columns=["A", "B", "C"]
    )
    sigma = pd.Series({"A": 0.02, "B": 0.04, "C": 0.05})

    bmp, p_value = bmp_statistic([abnormal], [sigma])

    window_len = 2
    car = abnormal.sum(axis=0).to_numpy()
    scar = car / (sigma.to_numpy() * np.sqrt(window_len))
    mean = scar.mean()
    se = scar.std(ddof=1) / np.sqrt(scar.size)
    bmp_ref = mean / se
    p_ref = float(2.0 * stats.t.sf(abs(bmp_ref), df=scar.size - 1))

    assert bmp == pytest.approx(bmp_ref, abs=1e-12)
    assert p_value == pytest.approx(p_ref, abs=1e-12)
