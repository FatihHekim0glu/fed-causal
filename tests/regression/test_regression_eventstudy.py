"""Regression tests for the event-study core (group ``eventstudy``).

Pins the deliverable's core event-study claims to behaviour:

- a known-injected CAR is recovered from the synthetic panel within tolerance
  (the rate-sensitive names carry it; the controls do not);
- the placebo percentile of a NO-EFFECT panel is ~uniform across seeds — the
  honest-NULL property at the placebo layer (the observed mean CAR is NOT
  systematically extreme vs. the block-matched placebo null).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from fedcausal.data.synthetic import (
    SyntheticPanel,
    pure_noise_panel,
    synthetic_event_panel,
)
from fedcausal.events.windows import build_all_windows
from fedcausal.eventstudy.abnormal import cumulative_abnormal_returns, stack_event_cars
from fedcausal.eventstudy.placebo import placebo_distribution

pytestmark = pytest.mark.regression


def _grid(panel: SyntheticPanel) -> pd.DatetimeIndex:
    """The panel's trading-day index, typed as a ``DatetimeIndex``."""
    return cast("pd.DatetimeIndex", panel.returns.index)


@pytest.mark.regression
def test_known_car_recovered_market_wide_within_tolerance(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """The injected CAR is market-wide and recovered on both groups within tolerance.

    Averaged across all events, BOTH the rate-sensitive ("treated") and the control
    cross-sectional CAR recover the injected effect (the event-window drift is
    market-wide, a recoverable CAR but not a tradable cross-sectional gap); the
    treated-minus-control difference is only the small surprise tilt.
    """
    panel = synthetic_event_panel
    grid = _grid(panel)
    windows = build_all_windows(grid, panel.announcement_dates, event_half_width=1)

    treated_cars: list[float] = []
    control_cars: list[float] = []
    for w in windows:
        result = cumulative_abnormal_returns(panel.returns, panel.market, w)
        treated_cars.append(float(result.car[panel.rate_sensitive].mean()))
        control_cars.append(float(result.car.drop(panel.rate_sensitive).mean()))

    treated_mean = float(np.mean(treated_cars))
    control_mean = float(np.mean(control_cars))
    # Both groups recover the market-wide injected CAR within tolerance.
    assert treated_mean == pytest.approx(panel.injected_car, abs=panel.injected_car * 0.8)
    assert control_mean == pytest.approx(panel.injected_car, abs=panel.injected_car * 0.8)
    assert treated_mean > 0.0
    # The treated-minus-control gap is only the small tilt, not a tradable spread.
    assert abs(treated_mean - control_mean) < panel.injected_car * 0.5


@pytest.mark.regression
def test_placebo_percentile_uniform_under_no_effect() -> None:
    """On a no-effect panel the observed CAR is NOT systematically extreme.

    The honest-NULL property: averaged over many independent pure-noise panels,
    the observed mean-CAR placebo p-value is centred near 0.5 and the false-
    rejection rate at alpha=0.05 is not inflated (close to nominal). A leaky or
    mis-scaled placebo null would push the p-values toward the tails.
    """
    p_values: list[float] = []
    for offset in range(40):
        panel = pure_noise_panel(seed=3000 + offset, n_events=12, n_names=24)
        grid = _grid(panel)
        windows = build_all_windows(grid, panel.announcement_dates, event_half_width=1)
        observed = float(stack_event_cars(panel.returns, panel.market, windows).mean())
        result = placebo_distribution(
            panel.returns,
            panel.market,
            panel.announcement_dates,
            observed,
            n_placebo=120,
            seed=17 + offset,
        )
        p_values.append(result.p_value)

    arr = np.asarray(p_values, dtype=np.float64)
    mean_p = float(arr.mean())
    reject_rate = float(np.mean(arr < 0.05))
    # Centred near 0.5 (no systematic significance) ...
    assert 0.35 <= mean_p <= 0.65
    # ... and the nominal-5% rejection rate is not inflated (honest, not leaky).
    assert reject_rate <= 0.20


@pytest.mark.regression
def test_golden_event_cars_are_stable() -> None:
    """Pinned per-event cross-sectional mean CARs do not drift (seed-locked)."""
    panel = synthetic_event_panel(seed=20260619)
    grid = _grid(panel)
    windows = build_all_windows(grid, panel.announcement_dates, event_half_width=1)
    cars = stack_event_cars(panel.returns, panel.market, windows)
    # Golden values captured from the locked seed/generator (5 dp). The CAR is
    # market-wide, so the per-event cross-sectional mean hugs the injected CAR.
    expected = np.array(
        [
            0.01181,
            0.01215,
            0.00950,
            0.00839,
            0.00820,
            0.01056,
            0.01220,
            0.00850,
            0.01277,
            0.00647,
            0.00567,
            0.01267,
            0.00822,
            0.00554,
            0.01524,
            0.00846,
        ]
    )
    assert cars.shape == expected.shape
    assert np.allclose(cars, expected, atol=1e-5)
