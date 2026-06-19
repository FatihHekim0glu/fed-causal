"""Property-based invariants for the FOMC calendar + windowing (group ``events``).

These encode the leakage guards the event layer must keep for ANY valid input:

- estimation and event windows are ALWAYS disjoint (no overlap), for any feasible
  ``(k, estimation_window, estimation_gap)`` geometry on any grid;
- ``build_all_windows`` never returns two windows that straddle (overlap);
- surprise classification depends ONLY on the sign of the at-announcement rate
  change — it is monotone and uses no future-revision magnitude;
- windowing is deterministic: identical inputs give byte-identical windows.

The placebo-exclusion and market-model-beta invariants live with their own
groups; this file owns the window-geometry and surprise-sign invariants.
"""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fedcausal._exceptions import InsufficientDataError
from fedcausal.events.calendar import classify_surprise
from fedcausal.events.windows import build_all_windows, build_windows

pytestmark = pytest.mark.property

# Window-geometry strategies (bounded to the library's documented ranges).
event_half_widths = st.integers(min_value=1, max_value=10)
estimation_windows = st.integers(min_value=2, max_value=200)
estimation_gaps = st.integers(min_value=1, max_value=20)
_GRID = pd.date_range("2015-01-01", periods=900, freq="B")


@settings(max_examples=200)
@given(
    event_pos=st.integers(min_value=0, max_value=len(_GRID) - 1),
    k=event_half_widths,
    est=estimation_windows,
    gap=estimation_gaps,
)
def test_estimation_and_event_windows_never_overlap(
    event_pos: int, k: int, est: int, gap: int
) -> None:
    """For any feasible geometry, the estimation and event windows are disjoint."""
    # ``k <= gap`` is required for the documented geometry to stay touch-free; let
    # Hypothesis only explore feasible geometries (the unit suite covers the raise).
    assume(k <= gap)
    ann = _GRID[event_pos].date()
    try:
        w = build_windows(_GRID, ann, event_half_width=k, estimation_window=est, estimation_gap=gap)
    except InsufficientDataError:
        # Not enough pre/post history at this grid position: skip this example.
        assume(False)
        return
    assert not w.overlaps
    assert w.estimation_end < w.event_start
    # the estimation window holds exactly ``est`` inclusive positions.
    assert w.estimation_end - w.estimation_start + 1 == est
    # the event window holds exactly ``2k + 1`` inclusive positions.
    assert w.event_end - w.event_start + 1 == 2 * k + 1


@settings(max_examples=150)
@given(
    positions=st.lists(
        st.integers(min_value=0, max_value=len(_GRID) - 1),
        min_size=0,
        max_size=12,
        unique=True,
    ),
    k=event_half_widths,
    gap=estimation_gaps,
)
def test_build_all_windows_are_mutually_non_straddling(
    positions: list[int], k: int, gap: int
) -> None:
    """No two returned windows ever overlap, and each is internally disjoint."""
    assume(k <= gap)
    dates = [_GRID[p].date() for p in positions]
    out = build_all_windows(
        _GRID, dates, event_half_width=k, estimation_window=60, estimation_gap=gap
    )
    # internally disjoint
    for w in out:
        assert not w.overlaps
    # chronologically ordered and pairwise non-overlapping event windows
    for earlier, later in pairwise(out):
        assert earlier.event_index < later.event_index
        assert later.event_start > earlier.event_end


@settings(max_examples=200)
@given(bps=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False))
def test_surprise_classification_is_sign_only(bps: float) -> None:
    """The label is a pure function of the sign of the at-announcement change."""
    label = classify_surprise(bps)
    if bps > 0.0:
        assert label == "hawkish"
    elif bps < 0.0:
        assert label == "dovish"
    else:
        assert label == "neutral"


@settings(max_examples=150)
@given(
    bps=st.floats(min_value=1e-6, max_value=1000.0, allow_nan=False),
    scale=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
)
def test_surprise_label_ignores_magnitude_only_sign(bps: float, scale: float) -> None:
    """Scaling a change by any positive factor preserves its (sign-driven) label.

    A larger or smaller positive change is still hawkish; the magnitude (which a
    future revision might alter) never changes the label — only the sign does.
    """
    assert classify_surprise(bps) == classify_surprise(bps * scale)
    assert classify_surprise(-bps) == classify_surprise(-bps * scale)


@settings(max_examples=100)
@given(
    event_pos=st.integers(min_value=200, max_value=len(_GRID) - 30),
    k=event_half_widths,
    est=st.integers(min_value=2, max_value=120),
    gap=estimation_gaps,
)
def test_windowing_is_deterministic(event_pos: int, k: int, est: int, gap: int) -> None:
    """Identical inputs always produce byte-identical windows (no hidden state)."""
    assume(k <= gap)
    ann = _GRID[event_pos].date()
    a = build_windows(_GRID, ann, event_half_width=k, estimation_window=est, estimation_gap=gap)
    b = build_windows(_GRID, ann, event_half_width=k, estimation_window=est, estimation_gap=gap)
    assert a == b
    assert a.to_dict() == b.to_dict()
