"""Expected returns and cumulative abnormal returns (CAR).

Implements the Brown-Warner (1985) / MacKinlay (1997) event-study core:

- ``market`` model: fit ``r_it = alpha_i + beta_i * r_mt + eps_it`` by OLS on the
  ESTIMATION WINDOW ONLY (pre-event), then form the abnormal return
  ``AR_it = r_it - (alpha_i + beta_i * r_mt)`` over the event window.
- ``mean_adjusted`` model: expected return is the estimation-window mean.
- ``CAR_i`` is the sum of ``AR_it`` over the event window ``[-k, +k]``.

LEAKAGE GUARD (enforced + tested): the model parameters are estimated on the
estimation window ONLY. Perturbing event-window returns leaves the fitted betas
byte-identical (a property test asserts this). The estimation and event windows
never overlap (see :mod:`fedcausal.events.windows`).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from fedcausal._typing import FloatArray, ModelKind
    from fedcausal.events.windows import EventWindows


@dataclass(frozen=True, slots=True)
class MarketModel:
    """Fitted per-name market-model parameters (estimation window only).

    Attributes
    ----------
    alpha:
        Per-name intercepts (index = ticker).
    beta:
        Per-name market slopes (index = ticker); all-zero for ``mean_adjusted``.
    sigma:
        Per-name estimation-window residual standard deviation (used to
        standardize abnormal returns in the BMP test).
    model:
        Which expected-return model was fit (``"market"`` or ``"mean_adjusted"``).
    n_obs:
        Number of estimation-window observations the fit used.
    """

    alpha: pd.Series
    beta: pd.Series
    sigma: pd.Series
    model: ModelKind
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the fitted parameters."""
        return {
            "alpha": {str(k): float(v) for k, v in self.alpha.items()},
            "beta": {str(k): float(v) for k, v in self.beta.items()},
            "sigma": {str(k): float(v) for k, v in self.sigma.items()},
            "model": self.model,
            "n_obs": int(self.n_obs),
        }


@dataclass(frozen=True, slots=True)
class CARResult:
    """Cumulative abnormal returns for one event across the cross-section.

    Attributes
    ----------
    car:
        Per-name CAR over the event window (index = ticker).
    abnormal:
        The per-name, per-day abnormal-return matrix over the event window
        (rows = event-relative day, columns = ticker).
    car_path:
        The cross-sectional mean cumulative abnormal return at each event-relative
        day (length ``2k + 1``), used for the CAR-path figure.
    model:
        The fitted market model used to form abnormal returns.
    """

    car: pd.Series
    abnormal: pd.DataFrame
    car_path: FloatArray
    model: MarketModel

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` (CAR + path only)."""
        return {
            "car": {str(k): float(v) for k, v in self.car.items()},
            "car_path": [float(x) for x in self.car_path],
            "model": self.model.to_dict(),
        }


def fit_market_model(
    returns: pd.DataFrame,
    market: pd.Series,
    windows: EventWindows,
    *,
    model: ModelKind = "market",
) -> MarketModel:
    """Fit the expected-return model on the ESTIMATION WINDOW ONLY.

    LEAKAGE GUARD: the OLS fit uses exclusively the estimation-window slice
    ``[estimation_start, estimation_end]`` of ``returns``/``market``. No
    event-window row is touched, so the fitted parameters are invariant to any
    perturbation of event-window returns (property-tested).

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series (regressor).
    windows:
        The leakage-safe windows for this event.
    model:
        ``"market"`` (alpha + beta * market) or ``"mean_adjusted"`` (mean only).

    Returns
    -------
    MarketModel
        The fitted per-name parameters and residual scale.

    Raises
    ------
    InsufficientDataError
        If the estimation window has too few observations.
    """
    raise NotImplementedError


def abnormal_returns(
    returns: pd.DataFrame,
    market: pd.Series,
    fitted: MarketModel,
    windows: EventWindows,
) -> pd.DataFrame:
    """Form per-name, per-day abnormal returns over the EVENT window.

    ``AR_it = r_it - (alpha_i + beta_i * r_mt)`` for the ``market`` model, or
    ``AR_it = r_it - mean_i`` for ``mean_adjusted``, evaluated on the event-window
    slice only.

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series.
    fitted:
        The model fitted on the estimation window.
    windows:
        The windows for this event.

    Returns
    -------
    pandas.DataFrame
        Abnormal returns over the event window (rows = event-relative day,
        columns = ticker).
    """
    raise NotImplementedError


def cumulative_abnormal_returns(
    returns: pd.DataFrame,
    market: pd.Series,
    windows: EventWindows,
    *,
    model: ModelKind = "market",
) -> CARResult:
    """Compute the cumulative abnormal return for one event (fit + AR + sum).

    Convenience wrapper: fits the model on the estimation window, forms abnormal
    returns over the event window, and cumulates them per name.

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series.
    windows:
        The leakage-safe windows for this event.
    model:
        The expected-return model.

    Returns
    -------
    CARResult
        Per-name CARs, the abnormal-return matrix, the cross-sectional CAR path,
        and the fitted model.
    """
    raise NotImplementedError


def stack_event_cars(
    returns: pd.DataFrame,
    market: pd.Series,
    windows_list: list[EventWindows],
    *,
    model: ModelKind = "market",
) -> np.ndarray:
    """Compute the cross-sectional mean CAR for each event in a list.

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series.
    windows_list:
        The leakage-safe windows for each event.
    model:
        The expected-return model.

    Returns
    -------
    numpy.ndarray
        A 1-D array of per-event cross-sectional mean CARs (length = number of
        events), the input to the HAC/placebo machinery.
    """
    raise NotImplementedError
