"""Event-study significance tests (cross-sectional t, BMP, HAC).

Three complementary significance statistics for cumulative abnormal returns:

- a plain **cross-sectional t-test** of the per-event (or per-name) CARs
  (Brown-Warner 1985);
- the **Boehmer-Musumeci-Poulsen (1991)** standardized-residual ("BMP") test,
  which standardizes each name's CAR by its estimation-window residual scale
  before averaging, making it robust to event-induced variance changes; and
- a **HAC / Newey-West** standard error of the mean CAR (reusing
  :mod:`fedcausal.evaluation.hac`) so serial correlation across events does not
  understate the standard error.

IMPORTANT: per the brief, a raw cross-sectional t-stat ALONE is NOT sufficient
evidence for the tradable verdict — the placebo null (see
:mod:`fedcausal.eventstudy.placebo`) is the PRIMARY significance source. These
statistics are reported alongside it.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from fedcausal._constants import DEFAULT_ALPHA

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@dataclass(frozen=True, slots=True)
class CARTestResult:
    """The battery of CAR significance statistics for one spec.

    Attributes
    ----------
    car_mean:
        The mean cumulative abnormal return across events (or names).
    t_stat:
        The plain cross-sectional t-statistic of the CARs.
    t_pvalue:
        The two-sided p-value of ``t_stat``.
    bmp_stat:
        The Boehmer-Musumeci-Poulsen standardized-residual statistic.
    bmp_pvalue:
        The two-sided p-value of ``bmp_stat``.
    hac_se:
        The Newey-West HAC standard error of ``car_mean``.
    hac_pvalue:
        The two-sided p-value of ``car_mean`` using the HAC standard error.
    n_obs:
        Number of CAR observations (events) the statistics are based on.
    """

    car_mean: float
    t_stat: float
    t_pvalue: float
    bmp_stat: float
    bmp_pvalue: float
    hac_se: float
    hac_pvalue: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the statistics."""
        return asdict(self)


def cross_sectional_t(cars: np.ndarray) -> tuple[float, float]:
    """Plain cross-sectional t-test of a vector of CARs (Brown-Warner 1985).

    Parameters
    ----------
    cars:
        A 1-D array of per-event (or per-name) CARs.

    Returns
    -------
    tuple[float, float]
        ``(t_stat, two_sided_pvalue)`` for ``H0: mean(cars) == 0``.

    Raises
    ------
    InsufficientDataError
        If fewer than two finite CARs are supplied.
    """
    raise NotImplementedError


def bmp_statistic(
    abnormal_by_event: list[pd.DataFrame],
    sigmas_by_event: list[pd.Series],
) -> tuple[float, float]:
    """Boehmer-Musumeci-Poulsen (1991) standardized cross-sectional test.

    Each name's event CAR is standardized by its estimation-window residual scale
    (and a forecast-error adjustment) to a standardized abnormal return (SAR);
    the BMP statistic is the cross-sectional mean SAR scaled by its own
    cross-sectional standard error, which is robust to event-induced variance
    inflation that biases the plain cross-sectional t.

    Parameters
    ----------
    abnormal_by_event:
        Per-event abnormal-return matrices (rows = event-relative day, columns =
        ticker).
    sigmas_by_event:
        Per-event, per-name estimation-window residual standard deviations.

    Returns
    -------
    tuple[float, float]
        ``(bmp_stat, two_sided_pvalue)``.

    Raises
    ------
    ValidationError
        If the two inputs are misaligned or empty.
    """
    raise NotImplementedError


def hac_car_test(
    cars: np.ndarray,
    *,
    lag: int | None = None,
) -> tuple[float, float, float]:
    """HAC / Newey-West test of the mean CAR (reuses :mod:`fedcausal.evaluation.hac`).

    Parameters
    ----------
    cars:
        A 1-D array of per-event CARs (ordered in time).
    lag:
        Bartlett lag truncation; ``None`` selects the Andrews rule.

    Returns
    -------
    tuple[float, float, float]
        ``(car_mean, hac_se, two_sided_pvalue)``.

    Raises
    ------
    InsufficientDataError
        If fewer than two finite CARs are supplied.
    """
    raise NotImplementedError


def run_car_tests(
    cars: np.ndarray,
    abnormal_by_event: list[pd.DataFrame],
    sigmas_by_event: list[pd.Series],
    *,
    alpha: float = DEFAULT_ALPHA,
    lag: int | None = None,
) -> CARTestResult:
    """Run the full CAR test battery (cross-sectional t + BMP + HAC).

    Parameters
    ----------
    cars:
        Per-event cross-sectional mean CARs.
    abnormal_by_event:
        Per-event abnormal-return matrices for the BMP test.
    sigmas_by_event:
        Per-event, per-name estimation-window residual scales for the BMP test.
    alpha:
        Significance level (recorded for downstream verdict use).
    lag:
        HAC lag truncation; ``None`` selects the Andrews rule.

    Returns
    -------
    CARTestResult
        The assembled statistics.
    """
    raise NotImplementedError
