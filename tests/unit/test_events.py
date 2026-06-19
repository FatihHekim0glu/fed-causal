"""Unit tests for the FOMC calendar + leakage-safe windowing (group ``events``).

Coverage:

- ``classify_surprise``: a hike is hawkish, a cut is dovish, no change is neutral
  (the sign rule uses ONLY the at-announcement rate change — no future revision).
- ``load_fomc_calendar``: builds chronologically-ordered, surprise-labelled events
  from the committed reference snapshot; the first meeting's predecessor is the
  committed pre-first target; date bounds clip inclusively; known liftoff/cut
  meetings carry the correct signed bps and labels.
- ``event_dates_frame``: a date-indexed, sorted-ascending view with the expected
  columns.
- windowing: the event window is ``[-k, +k]`` about the announcement position, the
  estimation window ends a strictly-positive gap before it and never overlaps;
  geometry/feasibility errors are raised, not silently truncated; building all
  windows drops straddling and infeasible events; weekend dates anchor forward.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fedcausal._exceptions import (
    EventCalendarError,
    InsufficientDataError,
    ValidationError,
    WindowOverlapError,
)
from fedcausal.events.calendar import (
    FOMCEvent,
    classify_surprise,
    event_dates_frame,
    load_fomc_calendar,
)
from fedcausal.events.calendar_data import (
    FOMC_ANNOUNCEMENTS,
    PRE_FIRST_TARGET_UPPER,
)
from fedcausal.events.windows import (
    EventWindows,
    assert_no_overlap,
    build_all_windows,
    build_windows,
)

# --------------------------------------------------------------------------- #
# classify_surprise                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("bps", "expected"),
    [
        (25.0, "hawkish"),
        (75.0, "hawkish"),
        (0.0, "neutral"),
        (-25.0, "dovish"),
        (-50.0, "dovish"),
    ],
)
def test_classify_surprise_sign_rule(bps: float, expected: str) -> None:
    """A hike is hawkish, a cut is dovish, no change is neutral."""
    assert classify_surprise(bps) == expected


@pytest.mark.unit
def test_classify_surprise_uses_only_the_signed_change() -> None:
    """The label depends solely on the sign of the at-announcement rate change.

    Tiny positive/negative perturbations flip the label deterministically; there
    is no dependence on any later-revised value (no future information).
    """
    assert classify_surprise(1e-9) == "hawkish"
    assert classify_surprise(-1e-9) == "dovish"
    assert classify_surprise(0.0) == "neutral"


# --------------------------------------------------------------------------- #
# load_fomc_calendar                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_calendar_returns_all_committed_events_in_order() -> None:
    """Every committed meeting is loaded, chronologically ordered, typed."""
    events = load_fomc_calendar()
    assert len(events) == len(FOMC_ANNOUNCEMENTS)
    assert all(isinstance(ev, FOMCEvent) for ev in events)
    dates = [ev.announcement_date for ev in events]
    assert dates == sorted(dates)
    # all three surprise signs are present across the committed span.
    assert {ev.surprise for ev in events} == {"hawkish", "dovish", "neutral"}


@pytest.mark.unit
def test_first_event_change_is_vs_committed_pre_first_target() -> None:
    """The first meeting's signed change is measured against the committed predecessor.

    The committed pre-first upper bound equals the first listed meeting's upper
    bound (0.25), so the first meeting is a no-change (neutral) event with 0 bps.
    """
    first = load_fomc_calendar()[0]
    first_iso, first_upper = FOMC_ANNOUNCEMENTS[0]
    assert first.announcement_date == date.fromisoformat(first_iso)
    expected_bps = (float(first_upper) - PRE_FIRST_TARGET_UPPER) * 100.0
    assert first.rate_change_bps == pytest.approx(expected_bps)
    assert first.surprise == classify_surprise(expected_bps)


@pytest.mark.unit
def test_known_liftoff_meeting_is_hawkish_25bps() -> None:
    """The 2022-03-16 liftoff (0.25 -> 0.50) is +25 bps, hawkish."""
    events = load_fomc_calendar()
    liftoff = next(ev for ev in events if ev.announcement_date == date(2022, 3, 16))
    assert liftoff.rate_change_bps == pytest.approx(25.0)
    assert liftoff.surprise == "hawkish"


@pytest.mark.unit
def test_known_first_cut_meeting_is_dovish_50bps() -> None:
    """The 2024-09-18 first cut (5.50 -> 5.00) is -50 bps, dovish."""
    events = load_fomc_calendar()
    cut = next(ev for ev in events if ev.announcement_date == date(2024, 9, 18))
    assert cut.rate_change_bps == pytest.approx(-50.0)
    assert cut.surprise == "dovish"


@pytest.mark.unit
def test_change_is_measured_against_the_previous_meeting_only() -> None:
    """Each meeting's bps change equals upper(this) - upper(previous), in bps.

    This is the PIT-honest definition: a meeting only knows the previous decision,
    never a future one. We replicate the sequence independently and compare.
    """
    events = load_fomc_calendar()
    prev_upper = PRE_FIRST_TARGET_UPPER
    for ev, (_iso, upper) in zip(events, FOMC_ANNOUNCEMENTS, strict=True):
        assert ev.rate_change_bps == pytest.approx((float(upper) - prev_upper) * 100.0)
        prev_upper = float(upper)


@pytest.mark.unit
def test_date_bounds_clip_inclusively() -> None:
    """``start``/``end`` clip the calendar inclusively."""
    bounded = load_fomc_calendar(start=date(2022, 1, 1), end=date(2022, 12, 31))
    assert bounded  # 2022 has scheduled meetings
    assert all(date(2022, 1, 1) <= ev.announcement_date <= date(2022, 12, 31) for ev in bounded)
    # a window with no meetings yields an empty list (not an error).
    assert load_fomc_calendar(start=date(2100, 1, 1)) == []


@pytest.mark.unit
def test_bounds_do_not_change_the_at_announcement_surprise_change() -> None:
    """Clipping the date range must not alter a kept event's signed change.

    The signed change is computed over the FULL committed sequence first; bounds
    are applied afterwards, so a meeting's surprise is identical whether or not
    earlier meetings were filtered out (no recomputation against a clipped
    predecessor — that would be a leakage-style artefact).
    """
    full = {ev.announcement_date: ev for ev in load_fomc_calendar()}
    bounded = load_fomc_calendar(start=date(2022, 6, 1), end=date(2022, 12, 31))
    for ev in bounded:
        assert ev.rate_change_bps == full[ev.announcement_date].rate_change_bps
        assert ev.surprise == full[ev.announcement_date].surprise


@pytest.mark.unit
def test_fomc_event_to_dict_is_json_serializable() -> None:
    """``FOMCEvent.to_dict`` emits an ISO date string + scalar fields."""
    ev = load_fomc_calendar()[0]
    payload = ev.to_dict()
    assert payload["announcement_date"] == ev.announcement_date.isoformat()
    assert set(payload) == {
        "announcement_date",
        "target_upper",
        "rate_change_bps",
        "surprise",
    }


# --------------------------------------------------------------------------- #
# event_dates_frame                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_event_dates_frame_is_sorted_date_indexed_view() -> None:
    """The frame is date-indexed, sorted ascending, with the expected columns."""
    events = load_fomc_calendar()
    frame = event_dates_frame(events)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert list(frame.columns) == ["target_upper", "rate_change_bps", "surprise"]
    assert len(frame) == len(events)
    # values line up with the events
    first = events[0]
    assert frame.iloc[0]["surprise"] == first.surprise
    assert float(frame.iloc[0]["target_upper"]) == pytest.approx(first.target_upper)


@pytest.mark.unit
def test_event_dates_frame_sorts_unordered_input() -> None:
    """An out-of-order event list is sorted ascending in the frame."""
    events = load_fomc_calendar()[:5]
    frame = event_dates_frame(list(reversed(events)))
    assert frame.index.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# load_fomc_calendar: malformed-data guards (honest, defensive)               #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_empty_committed_calendar_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty committed table raises ``EventCalendarError`` (no silent empty)."""
    import fedcausal.events.calendar as calendar_mod

    monkeypatch.setattr(calendar_mod, "FOMC_ANNOUNCEMENTS", ())
    with pytest.raises(EventCalendarError):
        load_fomc_calendar()


@pytest.mark.unit
def test_malformed_calendar_entry_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparseable date or non-numeric rate raises ``EventCalendarError``."""
    import fedcausal.events.calendar as calendar_mod

    monkeypatch.setattr(
        calendar_mod,
        "FOMC_ANNOUNCEMENTS",
        (("not-a-date", 0.25),),
    )
    with pytest.raises(EventCalendarError):
        load_fomc_calendar()


@pytest.mark.unit
def test_non_monotonic_calendar_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A calendar whose dates do not strictly increase raises ``EventCalendarError``."""
    import fedcausal.events.calendar as calendar_mod

    monkeypatch.setattr(
        calendar_mod,
        "FOMC_ANNOUNCEMENTS",
        (("2022-03-16", 0.50), ("2022-03-16", 0.75)),  # duplicate date
    )
    with pytest.raises(EventCalendarError):
        load_fomc_calendar()


# --------------------------------------------------------------------------- #
# windows: build_windows                                                      #
# --------------------------------------------------------------------------- #


def _grid(periods: int = 400, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="B")


@pytest.mark.unit
def test_build_windows_geometry_is_correct_and_disjoint() -> None:
    """The event/estimation spans match the documented integer geometry."""
    grid = _grid()
    ann = grid[200].date()
    k, est, gap = 2, 120, 10
    w = build_windows(grid, ann, event_half_width=k, estimation_window=est, estimation_gap=gap)
    assert isinstance(w, EventWindows)
    assert w.event_index == 200
    assert w.event_start == 200 - k
    assert w.event_end == 200 + k
    assert w.estimation_end == 200 - gap - 1
    assert w.estimation_start == w.estimation_end - est + 1
    # disjoint by construction
    assert not w.overlaps
    assert w.estimation_end < w.event_start
    assert_no_overlap(w)


@pytest.mark.unit
def test_estimation_window_has_the_requested_length() -> None:
    """The inclusive estimation span contains exactly ``estimation_window`` days."""
    grid = _grid()
    w = build_windows(grid, grid[200].date(), estimation_window=90, estimation_gap=5)
    length = w.estimation_end - w.estimation_start + 1
    assert length == 90


@pytest.mark.unit
def test_event_window_has_2k_plus_one_days() -> None:
    """The inclusive event span ``[-k, +k]`` spans ``2k + 1`` days."""
    grid = _grid()
    for k in (1, 3, 7):
        w = build_windows(grid, grid[200].date(), event_half_width=k, estimation_gap=10)
        assert w.event_end - w.event_start + 1 == 2 * k + 1


@pytest.mark.unit
def test_k_equals_gap_is_the_no_overlap_boundary() -> None:
    """``k == gap`` is feasible (windows touch-free); ``k == gap + 1`` overlaps."""
    grid = _grid()
    # k == gap: estimation_end = t-gap-1, event_start = t-k = t-gap -> disjoint.
    w = build_windows(
        grid, grid[200].date(), event_half_width=10, estimation_window=50, estimation_gap=10
    )
    assert not w.overlaps
    with pytest.raises(WindowOverlapError):
        build_windows(
            grid, grid[200].date(), event_half_width=3, estimation_window=50, estimation_gap=2
        )


@pytest.mark.unit
def test_weekend_announcement_anchors_to_next_trading_day() -> None:
    """A non-trading-day announcement anchors on the next available trading day."""
    grid = _grid()
    saturday = date(2020, 10, 10)
    assert saturday.weekday() == 5  # Saturday
    w = build_windows(grid, saturday, event_half_width=1)
    anchored = grid[w.event_index].date()
    assert anchored >= saturday
    assert grid[w.event_index].weekday() < 5  # a weekday


@pytest.mark.unit
def test_insufficient_pre_event_history_raises() -> None:
    """An event too close to the start (no full estimation window) raises."""
    grid = _grid()
    with pytest.raises(InsufficientDataError):
        build_windows(grid, grid[5].date(), estimation_window=120, estimation_gap=10)


@pytest.mark.unit
def test_insufficient_post_event_history_raises() -> None:
    """An event whose event window runs off the end of the grid raises."""
    grid = _grid()
    with pytest.raises(InsufficientDataError):
        build_windows(grid, grid[-1].date(), event_half_width=2)


@pytest.mark.unit
def test_off_grid_future_announcement_raises() -> None:
    """An announcement after the grid's last date raises ``EventCalendarError``."""
    grid = _grid()
    with pytest.raises(EventCalendarError):
        build_windows(grid, date(2100, 1, 1))


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"event_half_width": 0},
        {"event_half_width": 999},
        {"estimation_window": 1},
        {"estimation_gap": 0},
    ],
)
def test_build_windows_rejects_bad_geometry(kwargs: dict[str, int]) -> None:
    """Out-of-range geometry raises ``ValidationError`` before any layout."""
    grid = _grid()
    with pytest.raises(ValidationError):
        build_windows(grid, grid[200].date(), **kwargs)


@pytest.mark.unit
def test_build_windows_is_deterministic() -> None:
    """The same inputs always produce identical windows (no hidden state)."""
    grid = _grid()
    ann = grid[200].date()
    a = build_windows(grid, ann, event_half_width=2, estimation_window=100, estimation_gap=7)
    b = build_windows(grid, ann, event_half_width=2, estimation_window=100, estimation_gap=7)
    assert a == b
    assert a.to_dict() == b.to_dict()


@pytest.mark.unit
def test_event_windows_to_dict_is_json_serializable() -> None:
    """``EventWindows.to_dict`` emits an ISO date + integer positions."""
    grid = _grid()
    w = build_windows(grid, grid[200].date())
    payload = w.to_dict()
    assert payload["announcement_date"] == w.announcement_date.isoformat()
    assert payload["event_start"] == w.event_start
    assert payload["estimation_end"] == w.estimation_end


# --------------------------------------------------------------------------- #
# windows: assert_no_overlap (direct)                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_assert_no_overlap_raises_on_touching_windows() -> None:
    """A hand-built overlapping ``EventWindows`` is rejected."""
    bad = EventWindows(
        announcement_date=date(2020, 1, 1),
        event_index=100,
        estimation_start=0,
        estimation_end=99,  # touches event_start
        event_start=99,
        event_end=101,
    )
    assert bad.overlaps
    with pytest.raises(WindowOverlapError):
        assert_no_overlap(bad)


@pytest.mark.unit
def test_assert_no_overlap_passes_on_disjoint_windows() -> None:
    """A disjoint ``EventWindows`` passes silently."""
    good = EventWindows(
        announcement_date=date(2020, 1, 1),
        event_index=100,
        estimation_start=0,
        estimation_end=88,
        event_start=99,
        event_end=101,
    )
    assert not good.overlaps
    assert_no_overlap(good)  # no raise


# --------------------------------------------------------------------------- #
# windows: build_all_windows                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_build_all_windows_returns_one_per_feasible_event_in_order() -> None:
    """Well-spaced, feasible events each yield disjoint, ordered windows."""
    grid = _grid()
    ann = [grid[150].date(), grid[200].date(), grid[250].date()]
    out = build_all_windows(grid, ann)
    assert len(out) == 3
    starts = [w.event_index for w in out]
    assert starts == sorted(starts)
    for w in out:
        assert not w.overlaps


@pytest.mark.unit
def test_build_all_windows_sorts_unordered_dates() -> None:
    """Out-of-order announcement dates are processed chronologically."""
    grid = _grid()
    ann = [grid[250].date(), grid[150].date(), grid[200].date()]
    out = build_all_windows(grid, ann)
    indices = [w.event_index for w in out]
    assert indices == sorted(indices)


@pytest.mark.unit
def test_build_all_windows_drops_straddling_neighbours() -> None:
    """Two adjacent events whose windows would straddle: the later one is dropped."""
    grid = _grid()
    # positions 150 and 151 with k=1: windows [149,151] and [150,152] overlap.
    ann = [grid[150].date(), grid[151].date(), grid[250].date()]
    out = build_all_windows(grid, ann, event_half_width=1)
    assert len(out) == 2
    indices = [w.event_index for w in out]
    assert 150 in indices and 250 in indices
    assert 151 not in indices


@pytest.mark.unit
def test_build_all_windows_skips_infeasible_events() -> None:
    """An event with no usable pre-event history is omitted, not truncated."""
    grid = _grid()
    ann = [grid[5].date(), grid[200].date()]
    out = build_all_windows(grid, ann, estimation_window=120, estimation_gap=10)
    assert len(out) == 1
    assert out[0].event_index == 200


@pytest.mark.unit
def test_build_all_windows_validates_geometry_up_front() -> None:
    """Shared geometry is validated before any per-event work."""
    grid = _grid()
    with pytest.raises(ValidationError):
        build_all_windows(grid, [grid[200].date()], event_half_width=0)


@pytest.mark.unit
def test_build_all_windows_empty_input_is_empty_output() -> None:
    """No announcements -> no windows (not an error)."""
    grid = _grid()
    assert build_all_windows(grid, []) == []
