"""Plotly figure builders.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) and is
imported lazily inside each builder; importing this module has no side effects and
does not require Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/plots.py ({data, layout} shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def car_path_figure(
    car_path: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    *,
    event_half_width: int = 1,
    title: str = "Cumulative abnormal return around FOMC announcements",
) -> FigureDict:
    """Build the CAR-path figure (mean CAR with a confidence band).

    Plots the cross-sectional mean cumulative abnormal return at each event-
    relative day with a shaded confidence band, centred on the announcement day
    (event-relative day 0).

    Parameters
    ----------
    car_path:
        The mean cumulative abnormal return at each event-relative day (length
        ``2k + 1``).
    ci_lower, ci_upper:
        The lower/upper confidence-band bounds at each event-relative day.
    event_half_width:
        The half-width ``k`` (so the x-axis runs ``-k .. +k``).
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the CAR path with its band.
    """
    raise NotImplementedError


def placebo_histogram_figure(
    placebo_cars: np.ndarray,
    observed_car: float,
    *,
    percentile: float | None = None,
    title: str = "Placebo-date null distribution vs. observed CAR",
) -> FigureDict:
    """Build the placebo null-distribution histogram with the observed CAR marked.

    Plots the histogram of placebo-date cross-sectional mean CARs (the honest
    null) and draws a vertical marker at the observed CAR, so the reader can SEE
    that the observed effect sits inside (not in the tail of) the placebo null.

    Parameters
    ----------
    placebo_cars:
        The placebo-date null distribution of cross-sectional mean CARs.
    observed_car:
        The observed cross-sectional mean CAR (drawn as a vertical marker).
    percentile:
        Optional observed-CAR percentile to annotate.
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the placebo histogram.
    """
    raise NotImplementedError


def did_coefficient_figure(
    coef: float,
    ci_lower: float,
    ci_upper: float,
    *,
    title: str = "Difference-in-differences coefficient (treated minus control)",
) -> FigureDict:
    """Build the DiD coefficient figure (point estimate with a clustered-SE CI).

    Plots the ``treated x post`` interaction coefficient as a point with its
    clustered confidence interval, with a reference line at zero so the reader can
    see whether the heterogeneity is distinguishable from zero.

    Parameters
    ----------
    coef:
        The DiD interaction coefficient.
    ci_lower, ci_upper:
        The clustered confidence-interval bounds on ``coef``.
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the DiD coefficient.
    """
    raise NotImplementedError
