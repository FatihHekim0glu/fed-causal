"""Property-based invariants (Hypothesis) — the leakage guards.

The brief's property plan, encoded as invariants the compute kernels must keep:

- estimation/event windows never overlap and never straddle;
- placebo dates exclude every real event window;
- the fitted market-model betas are INVARIANT to any perturbation of
  event-window returns (the headline no-look-ahead guarantee);
- CAR additivity across disjoint sub-windows.

Authored sequentially once the compute kernels land — skipped in the scaffold.
The Hypothesis import is verified here so the dependency wiring is exercised.
"""

from __future__ import annotations

import pytest
from hypothesis import strategies as st

pytestmark = pytest.mark.property

_SCAFFOLD_REASON = "scaffold: property body authored once compute kernels land"

# Window-geometry strategies the authored tests will draw from.
event_half_widths = st.integers(min_value=1, max_value=10)
estimation_windows = st.integers(min_value=30, max_value=250)
estimation_gaps = st.integers(min_value=1, max_value=20)


def test_estimation_and_event_windows_never_overlap_or_straddle() -> None:
    """For any valid geometry, estimation and event windows are disjoint."""
    pytest.skip(_SCAFFOLD_REASON)


def test_placebo_dates_exclude_every_real_event_window() -> None:
    """Sampled placebo dates never fall inside a real event window."""
    pytest.skip(_SCAFFOLD_REASON)


def test_market_model_beta_invariant_to_event_window_perturbation() -> None:
    """Perturbing event-window returns leaves the fitted betas byte-identical."""
    pytest.skip(_SCAFFOLD_REASON)


def test_car_additivity_over_disjoint_subwindows() -> None:
    """CAR over ``[-k, +k]`` equals the sum of CARs over disjoint sub-windows."""
    pytest.skip(_SCAFFOLD_REASON)
