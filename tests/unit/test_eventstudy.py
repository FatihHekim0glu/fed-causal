"""Unit tests for the event-study core (group ``eventstudy``).

Covers the three group files:

- ``abnormal.py``: estimation-window-only market-model fit (leakage-safe), the
  ``mean_adjusted`` variant, abnormal returns, per-event CAR, the cross-sectional
  CAR path, and ``stack_event_cars``;
- ``tests.py``: the cross-sectional t, the Boehmer-Musumeci-Poulsen standardized
  statistic, the HAC / Newey-West mean-CAR test, and the assembled battery;
- ``placebo.py``: the placebo-date draw (leakage guard) and the placebo null
  distribution / observed-vs-placebo percentile.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from fedcausal._exceptions import (
    EventCalendarError,
    InsufficientDataError,
    ValidationError,
)
from fedcausal.data.synthetic import SyntheticPanel
from fedcausal.events.windows import EventWindows, build_all_windows, build_windows
from fedcausal.eventstudy.abnormal import (
    CARResult,
    MarketModel,
    abnormal_returns,
    cumulative_abnormal_returns,
    fit_market_model,
    stack_event_cars,
)
from fedcausal.eventstudy.placebo import (
    PlaceboResult,
    placebo_distribution,
    sample_placebo_dates,
)
from fedcausal.eventstudy.tests import (
    CARTestResult,
    bmp_statistic,
    cross_sectional_t,
    hac_car_test,
    run_car_tests,
)

# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _grid(panel: SyntheticPanel) -> pd.DatetimeIndex:
    """The panel's trading-day index, typed as a ``DatetimeIndex``."""
    return cast("pd.DatetimeIndex", panel.returns.index)


def _event_windows(panel: SyntheticPanel, k: int = 1) -> list[EventWindows]:
    return build_all_windows(_grid(panel), panel.announcement_dates, event_half_width=k)


# --------------------------------------------------------------------------- #
# abnormal.py — fit_market_model                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_fit_market_model_uses_estimation_window_only(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """The fit consumes exactly ``estimation_window`` pre-event rows."""
    panel = synthetic_event_panel
    w = build_windows(_grid(panel), panel.announcement_dates[4], estimation_window=120)
    fitted = fit_market_model(panel.returns, panel.market, w)
    assert isinstance(fitted, MarketModel)
    assert fitted.n_obs == 120
    assert fitted.model == "market"
    assert list(fitted.alpha.index) == list(panel.returns.columns)
    # sigma is a positive residual scale per name.
    assert (fitted.sigma.to_numpy() > 0).all()


@pytest.mark.unit
def test_fit_mean_adjusted_has_zero_beta(synthetic_event_panel: SyntheticPanel) -> None:
    """The ``mean_adjusted`` model carries an all-zero beta and a mean intercept."""
    panel = synthetic_event_panel
    w = build_windows(_grid(panel), panel.announcement_dates[4])
    fitted = fit_market_model(panel.returns, panel.market, w, model="mean_adjusted")
    assert fitted.model == "mean_adjusted"
    assert np.allclose(fitted.beta.to_numpy(), 0.0)
    # alpha is the estimation-window mean return per name.
    est = panel.returns.iloc[w.estimation_start : w.estimation_end + 1]
    assert np.allclose(fitted.alpha.to_numpy(), est.mean(axis=0).to_numpy())


@pytest.mark.unit
def test_fit_market_model_rejects_unknown_model(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """An unrecognized model kind raises ``ValidationError``."""
    panel = synthetic_event_panel
    w = build_windows(_grid(panel), panel.announcement_dates[4])
    with pytest.raises(ValidationError):
        fit_market_model(panel.returns, panel.market, w, model="garbage")  # type: ignore[arg-type]


@pytest.mark.unit
def test_fit_market_model_insufficient_estimation_window_raises(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """Too few estimation rows raises ``InsufficientDataError``."""
    panel = synthetic_event_panel
    w = build_windows(
        _grid(panel), panel.announcement_dates[4], estimation_window=2, estimation_gap=2
    )
    with pytest.raises(InsufficientDataError):
        fit_market_model(panel.returns, panel.market, w)


@pytest.mark.unit
def test_fit_market_model_to_dict_is_json_serializable(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """``MarketModel.to_dict`` emits scalar dicts + the model kind."""
    panel = synthetic_event_panel
    w = build_windows(_grid(panel), panel.announcement_dates[4])
    payload = fit_market_model(panel.returns, panel.market, w).to_dict()
    assert set(payload) == {"alpha", "beta", "sigma", "model", "n_obs"}
    assert isinstance(payload["n_obs"], int)
    assert all(isinstance(v, float) for v in payload["alpha"].values())


# --------------------------------------------------------------------------- #
# abnormal.py — abnormal_returns / CAR                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_abnormal_returns_shape_and_index(synthetic_event_panel: SyntheticPanel) -> None:
    """AR matrix is ``(2k+1) x n_names`` with event-relative day index."""
    panel = synthetic_event_panel
    k = 2
    w = build_windows(_grid(panel), panel.announcement_dates[4], event_half_width=k)
    fitted = fit_market_model(panel.returns, panel.market, w)
    ar = abnormal_returns(panel.returns, panel.market, fitted, w)
    assert ar.shape == (2 * k + 1, panel.returns.shape[1])
    assert list(ar.index) == [-2, -1, 0, 1, 2]
    assert list(ar.columns) == list(panel.returns.columns)


@pytest.mark.unit
def test_cumulative_abnormal_returns_recovers_market_wide_car(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """Both treated and control recover the market-wide CAR; treated adds a tilt.

    The injected CAR is MARKET-WIDE (a recoverable event-window drift, not a
    tradable cross-sectional gap), so both groups' cross-sectional CAR is
    materially positive; the rate-sensitive names carry only a small additional
    surprise tilt, so the treated-minus-control gap stays well below the CAR.
    """
    panel = synthetic_event_panel
    w = build_windows(_grid(panel), panel.announcement_dates[0], event_half_width=1)
    result = cumulative_abnormal_returns(panel.returns, panel.market, w)
    assert isinstance(result, CARResult)
    treated = float(result.car[panel.rate_sensitive].mean())
    control = float(result.car.drop(panel.rate_sensitive).mean())
    # The discriminator vs the old treated-only design: the control group ALSO
    # carries the market-wide CAR (under the old design it sat near zero). Both
    # groups recover the injected effect, comfortably positive.
    assert treated > panel.injected_car * 0.5
    assert control > panel.injected_car * 0.5
    # The treated-minus-control gap is only the small surprise tilt, but a SINGLE
    # event is too noisy to bound it tightly (per-name CAR noise dominates the
    # ~tilt-sized gap); the averaged-over-all-events gap bound lives in the
    # regression suite (test_known_car_recovered_market_wide_within_tolerance).


@pytest.mark.unit
def test_car_path_length_and_monotone_cumulation(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """The CAR path has ``2k+1`` points and its last value equals the mean CAR."""
    panel = synthetic_event_panel
    k = 1
    w = build_windows(_grid(panel), panel.announcement_dates[0], event_half_width=k)
    result = cumulative_abnormal_returns(panel.returns, panel.market, w)
    assert result.car_path.shape == (2 * k + 1,)
    # last cumulated cross-sectional mean == mean of per-name CARs.
    assert float(result.car_path[-1]) == pytest.approx(float(result.car.mean()))


@pytest.mark.unit
def test_car_additivity_over_disjoint_subwindows(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """CAR over ``[-k,+k]`` equals the sum of abnormal returns across all days."""
    panel = synthetic_event_panel
    w = build_windows(_grid(panel), panel.announcement_dates[3], event_half_width=3)
    result = cumulative_abnormal_returns(panel.returns, panel.market, w)
    # additivity: per-name CAR == sum over the per-day abnormal matrix.
    rebuilt = result.abnormal.sum(axis=0)
    assert np.allclose(result.car.to_numpy(), rebuilt.to_numpy())
    # and the day-block split is additive too.
    first_half = result.abnormal.loc[-3:0].sum(axis=0)
    second_half = result.abnormal.loc[1:3].sum(axis=0)
    assert np.allclose((first_half + second_half).to_numpy(), result.car.to_numpy())


@pytest.mark.unit
def test_stack_event_cars_one_per_event(synthetic_event_panel: SyntheticPanel) -> None:
    """``stack_event_cars`` returns one finite cross-sectional mean CAR per event."""
    panel = synthetic_event_panel
    windows = _event_windows(panel)
    cars = stack_event_cars(panel.returns, panel.market, windows)
    assert cars.shape == (len(windows),)
    assert np.isfinite(cars).all()
    # a known-effect panel has a positive average event CAR.
    assert float(cars.mean()) > 0.0


@pytest.mark.unit
def test_stack_event_cars_empty_raises(synthetic_event_panel: SyntheticPanel) -> None:
    """An empty windows list raises ``ValidationError``."""
    panel = synthetic_event_panel
    with pytest.raises(ValidationError):
        stack_event_cars(panel.returns, panel.market, [])


@pytest.mark.unit
def test_car_result_to_dict_is_json_serializable(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """``CARResult.to_dict`` emits scalar CARs + a float path + the model dict."""
    panel = synthetic_event_panel
    w = build_windows(_grid(panel), panel.announcement_dates[0])
    payload = cumulative_abnormal_returns(panel.returns, panel.market, w).to_dict()
    assert set(payload) == {"car", "car_path", "model"}
    assert all(isinstance(x, float) for x in payload["car_path"])
    assert payload["model"]["model"] == "market"


# --------------------------------------------------------------------------- #
# tests.py — cross-sectional t                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_cross_sectional_t_on_known_mean() -> None:
    """The t-stat of a shifted Gaussian sample is large and significant."""
    rng = np.random.default_rng(0)
    cars = rng.normal(0.01, 0.002, size=200)  # clearly non-zero mean
    t_stat, p_value = cross_sectional_t(cars)
    assert t_stat > 5.0
    assert p_value < 1e-6


@pytest.mark.unit
def test_cross_sectional_t_zero_mean_is_insignificant() -> None:
    """A zero-mean sample yields a small t-stat and a large p-value."""
    rng = np.random.default_rng(1)
    cars = rng.normal(0.0, 0.01, size=500)
    t_stat, p_value = cross_sectional_t(cars)
    assert abs(t_stat) < 3.0
    assert p_value > 0.01


@pytest.mark.unit
def test_cross_sectional_t_constant_sample_is_degenerate() -> None:
    """A zero-variance sample returns ``(0.0, 1.0)`` rather than dividing by zero."""
    t_stat, p_value = cross_sectional_t(np.full(10, 0.005))
    assert t_stat == 0.0
    assert p_value == 1.0


@pytest.mark.unit
def test_cross_sectional_t_too_few_obs_raises() -> None:
    """Fewer than two finite CARs raises ``InsufficientDataError``."""
    with pytest.raises(InsufficientDataError):
        cross_sectional_t(np.array([0.01]))


# --------------------------------------------------------------------------- #
# tests.py — BMP                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_bmp_misaligned_inputs_raise() -> None:
    """Mismatched / empty inputs raise ``ValidationError``."""
    abn = pd.DataFrame([[0.01, -0.01]], columns=["A", "B"])
    sig = pd.Series({"A": 0.02, "B": 0.03})
    with pytest.raises(ValidationError):
        bmp_statistic([abn, abn], [sig])  # length mismatch
    with pytest.raises(ValidationError):
        bmp_statistic([], [])


@pytest.mark.unit
def test_bmp_statistic_significant_on_known_effect(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """A real injected effect yields a significant BMP statistic."""
    panel = synthetic_event_panel
    windows = _event_windows(panel)
    abn = [cumulative_abnormal_returns(panel.returns, panel.market, w).abnormal for w in windows]
    sig = [
        cumulative_abnormal_returns(panel.returns, panel.market, w).model.sigma for w in windows
    ]
    bmp, p_value = bmp_statistic(abn, sig)
    assert bmp > 2.0
    assert p_value < 0.05


@pytest.mark.unit
def test_bmp_degenerate_zero_variance_is_neutral() -> None:
    """Identical standardized CARs (zero cross-sectional variance) -> ``(0.0, 1.0)``."""
    # Every name has the same CAR and the same sigma -> identical SCARs -> se=0.
    abn = pd.DataFrame([[0.01, 0.01, 0.01]], columns=["A", "B", "C"])
    sig = pd.Series({"A": 0.02, "B": 0.02, "C": 0.02})
    bmp, p_value = bmp_statistic([abn], [sig])
    assert bmp == 0.0
    assert p_value == 1.0


@pytest.mark.unit
def test_bmp_too_few_valid_observations_raises() -> None:
    """Fewer than two finite standardized CARs raises ``InsufficientDataError``.

    A single name (and zero-sigma names drop out of the standardization) leaves
    too few standardized observations for a cross-sectional statistic.
    """
    abn = pd.DataFrame([[0.01, 0.02]], columns=["A", "B"])
    sig = pd.Series({"A": 0.02, "B": 0.0})  # B has zero sigma -> dropped
    with pytest.raises(InsufficientDataError):
        bmp_statistic([abn], [sig])


# --------------------------------------------------------------------------- #
# tests.py — HAC                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_hac_car_test_returns_mean_se_pvalue() -> None:
    """The HAC test returns the sample mean, a positive SE, and a valid p-value."""
    rng = np.random.default_rng(2)
    cars = rng.normal(0.005, 0.002, size=120)
    mean, se, p_value = hac_car_test(cars)
    assert mean == pytest.approx(float(cars.mean()))
    assert se > 0.0
    assert 0.0 <= p_value <= 1.0
    assert p_value < 1e-3  # clearly non-zero mean


@pytest.mark.unit
def test_hac_car_test_too_few_obs_raises() -> None:
    """Fewer than two finite CARs raises ``InsufficientDataError``."""
    with pytest.raises(InsufficientDataError):
        hac_car_test(np.array([0.01]))


@pytest.mark.unit
def test_hac_car_test_zero_variance_is_neutral() -> None:
    """A constant CAR vector (zero HAC SE) returns a p-value of 1.0, not a divide."""
    mean, se, p_value = hac_car_test(np.full(20, 0.003))
    assert mean == pytest.approx(0.003)
    assert se == pytest.approx(0.0, abs=1e-12)
    assert p_value == 1.0


# --------------------------------------------------------------------------- #
# tests.py — battery                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_run_car_tests_assembles_full_battery(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """The battery assembles t / BMP / HAC into one result with ``n_obs``."""
    panel = synthetic_event_panel
    windows = _event_windows(panel)
    cars = stack_event_cars(panel.returns, panel.market, windows)
    abn = [cumulative_abnormal_returns(panel.returns, panel.market, w).abnormal for w in windows]
    sig = [
        cumulative_abnormal_returns(panel.returns, panel.market, w).model.sigma for w in windows
    ]
    result = run_car_tests(cars, abn, sig)
    assert isinstance(result, CARTestResult)
    assert result.n_obs == len(windows)
    assert result.car_mean == pytest.approx(float(cars.mean()))
    payload = result.to_dict()
    assert set(payload) >= {"car_mean", "t_stat", "bmp_stat", "hac_se", "hac_pvalue"}


@pytest.mark.unit
def test_run_car_tests_rejects_bad_alpha(synthetic_event_panel: SyntheticPanel) -> None:
    """An out-of-range ``alpha`` raises ``ValidationError`` before any compute."""
    panel = synthetic_event_panel
    windows = _event_windows(panel)
    cars = stack_event_cars(panel.returns, panel.market, windows)
    abn = [cumulative_abnormal_returns(panel.returns, panel.market, w).abnormal for w in windows]
    sig = [
        cumulative_abnormal_returns(panel.returns, panel.market, w).model.sigma for w in windows
    ]
    with pytest.raises(ValidationError):
        run_car_tests(cars, abn, sig, alpha=1.5)


# --------------------------------------------------------------------------- #
# placebo.py — sampling                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sample_placebo_dates_is_deterministic(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """The same seed reproduces the same placebo draw."""
    panel = synthetic_event_panel
    grid = _grid(panel)
    a = sample_placebo_dates(grid, panel.announcement_dates, n_placebo=50, event_half_width=1, seed=3)
    b = sample_placebo_dates(grid, panel.announcement_dates, n_placebo=50, event_half_width=1, seed=3)
    assert a == b
    assert len(a) == 50


@pytest.mark.unit
def test_sample_placebo_dates_excludes_event_windows(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """No placebo date falls inside any real event window (the leakage guard)."""
    panel = synthetic_event_panel
    grid = _grid(panel)
    k = 1
    dates = sample_placebo_dates(grid, panel.announcement_dates, n_placebo=400, event_half_width=k, seed=5)
    event_pos = {int(grid.searchsorted(pd.Timestamp(d), side="left")) for d in panel.announcement_dates}
    forbidden = {p + off for p in event_pos for off in range(-(k + 1), k + 2)}
    placebo_pos = [int(grid.searchsorted(pd.Timestamp(d), side="left")) for d in dates]
    assert all(p not in forbidden for p in placebo_pos)


@pytest.mark.unit
def test_sample_placebo_dates_rejects_nonpositive_n() -> None:
    """``n_placebo < 1`` raises ``EventCalendarError``."""
    grid = pd.date_range("2020-01-01", periods=400, freq="B")
    with pytest.raises(EventCalendarError):
        sample_placebo_dates(grid, [], n_placebo=0, event_half_width=1)


@pytest.mark.unit
def test_sample_placebo_dates_no_eligible_dates_raises() -> None:
    """A grid where every position is forbidden raises ``EventCalendarError``."""
    grid = pd.date_range("2020-01-01", periods=130, freq="B")
    # estimation_window=120 + gap=10 + k=1 burn-in already covers the whole grid.
    with pytest.raises(EventCalendarError):
        sample_placebo_dates(
            grid, [], n_placebo=10, event_half_width=1, estimation_window=120, estimation_gap=10
        )


# --------------------------------------------------------------------------- #
# placebo.py — distribution                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_placebo_distribution_fields_and_bounds(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """The placebo result has bounded percentile / p-value and a non-empty null."""
    panel = synthetic_event_panel
    windows = _event_windows(panel)
    cars = stack_event_cars(panel.returns, panel.market, windows)
    result = placebo_distribution(
        panel.returns, panel.market, panel.announcement_dates, float(cars.mean()), n_placebo=120, seed=7
    )
    assert isinstance(result, PlaceboResult)
    assert 0.0 <= result.percentile <= 100.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_placebo > 0
    assert result.placebo_cars.size == result.n_placebo
    # to_dict drops the bulk array and keeps the scalars.
    payload = result.to_dict()
    assert "placebo_cars" not in payload
    assert set(payload) == {"observed_car", "percentile", "p_value", "n_placebo"}


@pytest.mark.unit
def test_placebo_distribution_is_seed_reproducible(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """The same seed reproduces the same placebo null and percentile."""
    panel = synthetic_event_panel
    windows = _event_windows(panel)
    obs = float(stack_event_cars(panel.returns, panel.market, windows).mean())
    a = placebo_distribution(panel.returns, panel.market, panel.announcement_dates, obs, n_placebo=80, seed=9)
    b = placebo_distribution(panel.returns, panel.market, panel.announcement_dates, obs, n_placebo=80, seed=9)
    assert a.p_value == b.p_value
    assert np.array_equal(a.placebo_cars, b.placebo_cars)


@pytest.mark.unit
def test_placebo_percentile_grows_with_observed_extremity(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """A larger observed CAR sits higher in the placebo distribution."""
    panel = synthetic_event_panel
    windows = _event_windows(panel)
    base = float(stack_event_cars(panel.returns, panel.market, windows).mean())
    small = placebo_distribution(
        panel.returns, panel.market, panel.announcement_dates, base * 0.1, n_placebo=150, seed=4
    )
    large = placebo_distribution(
        panel.returns, panel.market, panel.announcement_dates, base * 5.0, n_placebo=150, seed=4
    )
    assert large.percentile >= small.percentile
    assert large.p_value <= small.p_value


@pytest.mark.unit
def test_placebo_distribution_single_event_block(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """With a single real event the placebo draws are per-anchor (no block averaging).

    A one-event family makes each placebo draw a single anchor CAR (the block size
    is one), so the null has exactly ``n_placebo`` entries.
    """
    panel = synthetic_event_panel
    one_event = panel.announcement_dates[:1]
    windows = _event_windows(panel)
    obs = float(stack_event_cars(panel.returns, panel.market, windows[:1]).mean())
    result = placebo_distribution(
        panel.returns, panel.market, one_event, obs, n_placebo=60, seed=2
    )
    assert result.n_placebo == 60
    assert 0.0 <= result.p_value <= 1.0
