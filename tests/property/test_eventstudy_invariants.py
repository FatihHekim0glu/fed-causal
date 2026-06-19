"""Property-based invariants for the event-study core (group ``eventstudy``).

The headline no-look-ahead guarantees, encoded as Hypothesis invariants:

- the fitted market-model betas (and alpha/sigma) are BYTE-IDENTICAL to any
  perturbation of event-window returns (the estimation window is the only input);
- sampled placebo dates never fall inside a real event window (the leakage guard);
- per-name CAR is additive across disjoint day sub-windows.
"""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fedcausal.data.synthetic import SyntheticPanel, synthetic_event_panel
from fedcausal.events.windows import build_windows
from fedcausal.eventstudy.abnormal import (
    abnormal_returns,
    cumulative_abnormal_returns,
    fit_market_model,
)
from fedcausal.eventstudy.placebo import sample_placebo_dates

pytestmark = pytest.mark.property

_SEED = 20260619


@lru_cache(maxsize=1)
def _panel() -> SyntheticPanel:
    """A single deterministic panel shared across the property cases (cached)."""
    return synthetic_event_panel(seed=_SEED, n_events=12, n_names=24)


def _grid(panel: SyntheticPanel) -> pd.DatetimeIndex:
    """The panel's trading-day index, typed as a ``DatetimeIndex``."""
    return cast("pd.DatetimeIndex", panel.returns.index)


# Geometry strategies kept inside the panel's feasible region.
event_half_widths = st.integers(min_value=1, max_value=4)
estimation_windows = st.integers(min_value=30, max_value=120)
estimation_gaps = st.integers(min_value=5, max_value=15)
event_choices = st.integers(min_value=2, max_value=9)  # interior events only


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    event_idx=event_choices,
    k=event_half_widths,
    est=estimation_windows,
    gap=estimation_gaps,
    bump=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_market_model_beta_invariant_to_event_window_perturbation(
    event_idx: int, k: int, est: int, gap: int, bump: float
) -> None:
    """Perturbing event-window returns leaves the fitted parameters byte-identical."""
    panel = _panel()
    grid = _grid(panel)
    ann = panel.announcement_dates[event_idx]
    w = build_windows(grid, ann, event_half_width=k, estimation_window=est, estimation_gap=gap)

    fitted_a = fit_market_model(panel.returns, panel.market, w)

    perturbed = panel.returns.copy()
    perturbed.iloc[w.event_start : w.event_end + 1] += bump
    fitted_b = fit_market_model(perturbed, panel.market, w)

    # Byte-identical: the estimation slice is untouched, so OLS is unchanged.
    assert np.array_equal(fitted_a.alpha.to_numpy(), fitted_b.alpha.to_numpy())
    assert np.array_equal(fitted_a.beta.to_numpy(), fitted_b.beta.to_numpy())
    assert np.array_equal(fitted_a.sigma.to_numpy(), fitted_b.sigma.to_numpy())


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    k=event_half_widths,
    n_placebo=st.integers(min_value=10, max_value=200),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_placebo_dates_exclude_every_real_event_window(
    k: int, n_placebo: int, seed: int
) -> None:
    """Sampled placebo dates never fall inside a real event window (plus buffer)."""
    panel = _panel()
    grid = _grid(panel)
    dates = sample_placebo_dates(
        grid, panel.announcement_dates, n_placebo=n_placebo, event_half_width=k, seed=seed
    )
    event_pos = {
        int(grid.searchsorted(pd.Timestamp(d), side="left")) for d in panel.announcement_dates
    }
    forbidden = {p + off for p in event_pos for off in range(-(k + 1), k + 2)}
    placebo_pos = [int(grid.searchsorted(pd.Timestamp(d), side="left")) for d in dates]
    assert all(p not in forbidden for p in placebo_pos)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(event_idx=event_choices, k=event_half_widths, split=st.integers(min_value=0, max_value=3))
def test_car_additivity_over_disjoint_subwindows(
    event_idx: int, k: int, split: int
) -> None:
    """CAR over ``[-k,+k]`` equals the sum of CARs over two disjoint day blocks."""
    panel = _panel()
    grid = _grid(panel)
    ann = panel.announcement_dates[event_idx]
    w = build_windows(grid, ann, event_half_width=k)
    fitted = fit_market_model(panel.returns, panel.market, w)
    ar = abnormal_returns(panel.returns, panel.market, fitted, w)

    full = ar.sum(axis=0).to_numpy()
    # Split the event-relative day index at an interior boundary.
    days = list(ar.index)
    cut = min(split, len(days) - 1)
    left = ar.iloc[: cut + 1].sum(axis=0).to_numpy()
    right = ar.iloc[cut + 1 :].sum(axis=0).to_numpy()
    assert np.allclose(left + right, full, atol=1e-12)

    # And the convenience wrapper agrees with the explicit sum.
    result = cumulative_abnormal_returns(panel.returns, panel.market, w)
    assert np.allclose(result.car.to_numpy(), full, atol=1e-12)
