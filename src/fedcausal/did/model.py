"""Difference-in-differences on rate-sensitive vs. control names.

The DiD design contrasts the abnormal-return response of **rate-sensitive**
("treated") names with **rate-insensitive** ("control") names around **hawkish**
vs. **dovish** surprises. The interaction coefficient ``treated x post`` measures
how much MORE rate-sensitive names move than controls, conditional on the
surprise sign.

Standard errors are CLUSTERED (by event, optionally two-way by event and name) so
the inference is robust to within-event cross-sectional correlation and serial
correlation across events.

FRAMING (the honest deliverable): the DiD coefficient is reported as DESCRIPTIVE
cross-sectional HETEROGENEITY — rate-sensitive names mechanically move more — NOT
as a tradable profit claim. Whether that heterogeneity translates into a
net-of-cost tradable spread is decided (negatively, by default) by
:mod:`fedcausal.evaluation.verdict`.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from fedcausal._constants import DEFAULT_ALPHA

if TYPE_CHECKING:
    import pandas as pd

    from fedcausal._typing import SurpriseLabel


@dataclass(frozen=True, slots=True)
class DiDResult:
    """The difference-in-differences interaction estimate (clustered SEs).

    Attributes
    ----------
    coef:
        The ``treated x post`` interaction coefficient — the descriptive
        treated-minus-control differential response.
    std_error:
        The clustered standard error of ``coef``.
    t_stat:
        ``coef / std_error``.
    p_value:
        The two-sided p-value of ``coef`` under the clustered SE.
    n_obs:
        Number of (name x period) observations in the regression.
    n_clusters:
        Number of clusters (events) the SE is based on.
    cluster:
        The clustering scheme used (``"event"`` or ``"event+name"``).
    """

    coef: float
    std_error: float
    t_stat: float
    p_value: float
    n_obs: int
    n_clusters: int
    cluster: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the DiD estimate."""
        return asdict(self)


def build_did_panel(
    abnormal_by_event: list[pd.DataFrame],
    surprises: list[SurpriseLabel],
    rate_sensitive: list[str],
    *,
    treated_surprise: SurpriseLabel = "hawkish",
    control_surprise: SurpriseLabel = "dovish",
) -> pd.DataFrame:
    """Assemble the long-format DiD regression panel.

    Stacks per-event abnormal returns into a tidy frame with columns
    ``y`` (abnormal return), ``treated`` (1 if rate-sensitive), ``post`` (1 if the
    surprise is the ``treated_surprise``), the ``treated:post`` interaction, an
    ``event`` cluster id, and a ``name`` id.

    Parameters
    ----------
    abnormal_by_event:
        Per-event abnormal-return matrices (rows = event-relative day, columns =
        ticker).
    surprises:
        The surprise label per event (parallel to ``abnormal_by_event``).
    rate_sensitive:
        The tickers designated rate-sensitive (treated).
    treated_surprise, control_surprise:
        Which surprise signs define the ``post`` contrast.

    Returns
    -------
    pandas.DataFrame
        The long-format DiD panel.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or empty.
    """
    raise NotImplementedError


def estimate_did(
    panel: pd.DataFrame,
    *,
    cluster: str = "event",
    alpha: float = DEFAULT_ALPHA,
) -> DiDResult:
    """Estimate the DiD interaction with CLUSTERED standard errors.

    Fits ``y ~ treated + post + treated:post`` by OLS and computes
    cluster-robust standard errors (by ``event``, or two-way by ``event`` and
    ``name``) so inference accounts for within-event and across-event correlation.

    Parameters
    ----------
    panel:
        The long-format DiD panel from :func:`build_did_panel`.
    cluster:
        ``"event"`` (one-way) or ``"event+name"`` (two-way) clustering.
    alpha:
        Significance level (recorded for downstream verdict use).

    Returns
    -------
    DiDResult
        The interaction coefficient and clustered inference.

    Raises
    ------
    ValidationError
        If ``panel`` is missing required columns or ``cluster`` is unknown.
    """
    raise NotImplementedError
