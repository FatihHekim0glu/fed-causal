"""Estimation- and event-window construction (leakage-safe).

For each FOMC announcement this module carves two disjoint windows on the
trading-day grid:

- the **estimation window** ``[t - gap - estimation_window, t - gap - 1]`` (the
  pre-event history the market model is fit on), and
- the **event window** ``[t - k, t + k]`` (the half-width-``k`` window over which
  the cumulative abnormal return is measured).

LEAKAGE GUARDS (enforced here, tested in ``tests/property``):

- The estimation and event windows NEVER overlap (a positive ``gap`` separates
  them), so no event-window return can leak into the fitted betas.
- An event window never straddles an adjacent event's window.
- Windows are clipped to the available trading-day grid; an event without enough
  pre-event history is reported, never silently truncated into the event window.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from fedcausal._constants import (
    DEFAULT_ESTIMATION_GAP,
    DEFAULT_ESTIMATION_WINDOW,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class EventWindows:
    """The disjoint estimation/event windows for a single FOMC event.

    Attributes
    ----------
    announcement_date:
        The event timestamp (FOMC announcement date).
    event_index:
        The integer position of the announcement day on the trading-day grid.
    estimation_start, estimation_end:
        Inclusive integer positions bounding the pre-event estimation window.
    event_start, event_end:
        Inclusive integer positions bounding the event window ``[-k, +k]``.
    """

    announcement_date: date
    event_index: int
    estimation_start: int
    estimation_end: int
    event_start: int
    event_end: int

    @property
    def overlaps(self) -> bool:
        """Whether the estimation and event windows touch (must always be ``False``)."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of these windows."""
        payload = asdict(self)
        payload["announcement_date"] = self.announcement_date.isoformat()
        return payload


def build_windows(
    grid: pd.DatetimeIndex,
    announcement_date: date,
    *,
    event_half_width: int = 1,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap: int = DEFAULT_ESTIMATION_GAP,
) -> EventWindows:
    """Build the disjoint estimation/event windows for one announcement.

    Locates ``announcement_date`` (or the next available trading day at/after it)
    on ``grid`` and lays out the event window ``[-k, +k]`` and the estimation
    window ``[t - gap - estimation_window, t - gap - 1]`` as integer positions.

    LEAKAGE GUARD: a strictly positive ``estimation_gap`` guarantees the
    estimation window ends before the event window begins, so the two are
    disjoint by construction.

    Parameters
    ----------
    grid:
        The sorted trading-day index of the panel.
    announcement_date:
        The FOMC announcement date to anchor the windows.
    event_half_width:
        The half-width ``k`` of the event window ``[-k, +k]``.
    estimation_window:
        The length (trading days) of the pre-event estimation window.
    estimation_gap:
        The strictly positive gap (trading days) between the estimation window
        and the event window.

    Returns
    -------
    EventWindows
        The integer-position windows for this event.

    Raises
    ------
    ValidationError
        If ``event_half_width``, ``estimation_window`` or ``estimation_gap`` are
        out of range.
    WindowOverlapError
        If the requested geometry would overlap or straddle.
    InsufficientDataError
        If there is not enough pre-event history on the grid.
    """
    raise NotImplementedError


def assert_no_overlap(windows: EventWindows) -> None:
    """Assert that an :class:`EventWindows` has disjoint estimation/event spans.

    Parameters
    ----------
    windows:
        The windows to check.

    Raises
    ------
    WindowOverlapError
        If ``estimation_end >= event_start`` (the windows touch or overlap).
    """
    raise NotImplementedError


def build_all_windows(
    grid: pd.DatetimeIndex,
    announcement_dates: list[date],
    *,
    event_half_width: int = 1,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap: int = DEFAULT_ESTIMATION_GAP,
) -> list[EventWindows]:
    """Build leakage-safe windows for every announcement, skipping infeasible ones.

    LEAKAGE GUARD: in addition to per-event no-overlap, consecutive event windows
    are checked for straddling; an event whose window would straddle a neighbour
    or lacks pre-event history is omitted rather than silently truncated.

    Parameters
    ----------
    grid:
        The sorted trading-day index of the panel.
    announcement_dates:
        The FOMC announcement dates.
    event_half_width:
        The half-width ``k`` of each event window.
    estimation_window:
        The pre-event estimation-window length.
    estimation_gap:
        The strictly positive estimation/event gap.

    Returns
    -------
    list[EventWindows]
        Feasible, mutually non-straddling windows in chronological order.
    """
    raise NotImplementedError
