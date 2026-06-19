"""FOMC event calendar and announcement-time surprise classification.

This module turns the committed reference calendar
(:mod:`fedcausal.events.calendar_data`) into a typed, date-indexed table of FOMC
events, each tagged with a ``hawkish``/``dovish``/``neutral`` surprise label.

LEAKAGE GUARD: the surprise label is derived ONLY from the sign of the realized
change in the federal funds target-rate upper bound from the previous meeting to
this one — information embodied in the decision itself, available AT the
announcement. No future revision, no pre-announcement signal, and the event
timestamp is the announcement DATE.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from fedcausal._typing import SurpriseLabel


@dataclass(frozen=True, slots=True)
class FOMCEvent:
    """A single FOMC announcement with its at-announcement surprise label.

    Attributes
    ----------
    announcement_date:
        The public statement (announcement) date — the event timestamp. There is
        no pre-announcement signal; the study windows are anchored here.
    target_upper:
        The federal funds target-rate upper bound (percent) in effect right
        after the decision.
    rate_change_bps:
        The change in the target upper bound from the previous meeting, in basis
        points (signed). The surprise sign is the sign of this number.
    surprise:
        ``"hawkish"`` (a hike), ``"dovish"`` (a cut), or ``"neutral"`` (no
        change). Derived only from ``rate_change_bps`` — information available at
        the announcement.
    """

    announcement_date: date
    target_upper: float
    rate_change_bps: float
    surprise: SurpriseLabel

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this event."""
        payload = asdict(self)
        payload["announcement_date"] = self.announcement_date.isoformat()
        return payload


def classify_surprise(rate_change_bps: float) -> SurpriseLabel:
    """Classify a meeting's surprise from the signed target-rate change.

    LEAKAGE GUARD: uses ONLY the realized rate change embodied in the decision
    (available at the announcement). A hike (``> 0``) is ``"hawkish"``, a cut
    (``< 0``) is ``"dovish"``, and no change (``== 0``) is ``"neutral"``.

    Parameters
    ----------
    rate_change_bps:
        The signed change in the federal funds target upper bound, in basis
        points.

    Returns
    -------
    SurpriseLabel
        The surprise label.
    """
    raise NotImplementedError


def load_fomc_calendar(
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[FOMCEvent]:
    """Load the committed FOMC calendar as typed, surprise-labelled events.

    Reads the committed public announcement dates and post-meeting target-rate
    snapshot from :mod:`fedcausal.events.calendar_data`, computes each meeting's
    signed rate change versus its predecessor, and classifies the surprise.

    Parameters
    ----------
    start, end:
        Optional inclusive date bounds; events outside the range are dropped.

    Returns
    -------
    list[FOMCEvent]
        Chronologically ordered FOMC events with surprise labels.

    Raises
    ------
    EventCalendarError
        If the committed calendar is empty or malformed.
    """
    raise NotImplementedError


def event_dates_frame(events: list[FOMCEvent]) -> pd.DataFrame:
    """Return a date-indexed DataFrame view of FOMC events for joins.

    Parameters
    ----------
    events:
        The FOMC events to tabulate.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``announcement_date`` with ``target_upper``,
        ``rate_change_bps`` and ``surprise`` columns, sorted ascending.
    """
    raise NotImplementedError
