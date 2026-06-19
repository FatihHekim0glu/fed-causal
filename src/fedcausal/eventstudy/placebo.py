"""Placebo-date randomization — the PRIMARY significance source.

The honest significance of the observed CAR is its percentile against a null
distribution built by re-running the SAME event study on random NON-event dates:

1. Draw ``n_placebo`` placebo "announcement" dates uniformly from the trading-day
   grid, EXCLUDING every real event window (plus a buffer) so no real effect can
   contaminate the null.
2. For each placebo date, build leakage-safe windows, fit the estimation-window
   market model, and compute the cross-sectional mean CAR.
3. The observed CAR's percentile within this placebo distribution is the honest
   significance: a real, exploitable effect sits far in the tail; noise sits near
   the middle (the percentile is ~uniform).

LEAKAGE GUARD (tested): placebo dates exclude all real event windows. On a
no-effect panel the observed-CAR percentile is ~uniform.

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
    import numpy as np
    import pandas as pd

    from fedcausal._typing import ModelKind


@dataclass(frozen=True, slots=True)
class PlaceboResult:
    """The placebo null distribution and the observed CAR's place within it.

    Attributes
    ----------
    observed_car:
        The observed cross-sectional mean CAR over the real events.
    placebo_cars:
        The null distribution of cross-sectional mean CARs from placebo dates.
    percentile:
        The percentile (0-100) of ``abs(observed_car)`` within the placebo
        distribution of ``abs`` CARs — the honest two-sided significance.
    p_value:
        ``1 - percentile/100`` (the placebo tail probability); the PRIMARY
        significance figure for the verdict.
    n_placebo:
        The number of valid placebo draws actually used.
    """

    observed_car: float
    placebo_cars: np.ndarray
    percentile: float
    p_value: float
    n_placebo: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` (excludes the full draw array)."""
        payload = asdict(self)
        payload.pop("placebo_cars", None)
        payload["observed_car"] = float(self.observed_car)
        payload["percentile"] = float(self.percentile)
        payload["p_value"] = float(self.p_value)
        payload["n_placebo"] = int(self.n_placebo)
        return payload


def sample_placebo_dates(
    grid: pd.DatetimeIndex,
    event_dates: list[date],
    *,
    n_placebo: int,
    event_half_width: int,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap: int = DEFAULT_ESTIMATION_GAP,
    seed: int = 7,
) -> list[date]:
    """Draw placebo dates that EXCLUDE every real event window (leakage guard).

    Builds the set of forbidden trading days (every real event window plus a
    buffer, and the leading burn-in that lacks pre-event history) and samples
    ``n_placebo`` dates uniformly from the remaining eligible days, seeded.

    Parameters
    ----------
    grid:
        The sorted trading-day index of the panel.
    event_dates:
        The real FOMC announcement dates to exclude.
    n_placebo:
        Number of placebo dates to draw.
    event_half_width:
        The event half-width ``k`` (defines each forbidden window's size).
    estimation_window:
        Estimation-window length (to exclude the leading burn-in).
    estimation_gap:
        The estimation/event gap.
    seed:
        Master RNG seed for the draw.

    Returns
    -------
    list[date]
        The sampled placebo dates (all outside every real event window).

    Raises
    ------
    EventCalendarError
        If too few eligible non-event dates remain to draw ``n_placebo``.
    """
    raise NotImplementedError


def placebo_distribution(
    returns: pd.DataFrame,
    market: pd.Series,
    event_dates: list[date],
    observed_car: float,
    *,
    n_placebo: int,
    event_half_width: int = 1,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap: int = DEFAULT_ESTIMATION_GAP,
    model: ModelKind = "market",
    seed: int = 7,
) -> PlaceboResult:
    """Build the placebo null and locate the observed CAR's percentile.

    Re-runs the cross-sectional mean CAR on each placebo date (using the same
    leakage-safe windowing and estimation-window-only model) and returns the
    observed CAR's two-sided percentile/p-value within that null.

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series.
    event_dates:
        The real FOMC announcement dates (excluded from the placebo draw).
    observed_car:
        The observed cross-sectional mean CAR over the real events.
    n_placebo:
        Number of placebo draws.
    event_half_width:
        The event half-width ``k``.
    estimation_window:
        Estimation-window length.
    estimation_gap:
        The estimation/event gap.
    model:
        The expected-return model.
    seed:
        Master RNG seed.

    Returns
    -------
    PlaceboResult
        The null distribution and the observed CAR's percentile/p-value.
    """
    raise NotImplementedError
