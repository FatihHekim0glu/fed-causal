"""Synthetic event panel with a KNOWN injected effect (ground truth).

The deployed default and the entire test suite run on a deterministic synthetic
event panel so the placebo/HAC/DiD machinery is validated against ground truth:

- ``synthetic_event_panel`` injects a KNOWN cumulative abnormal return (CAR) into
  the event windows of rate-sensitive names, on top of a one-factor market model
  plus idiosyncratic noise. A correct event study must recover the injected CAR
  within tolerance.
- ``pure_noise_panel`` injects NO effect, so a correct, honest pipeline must
  return ``fed_effect_is_tradable=False`` and a placebo percentile that is
  ~uniform.

Everything is seeded via :func:`fedcausal._rng.make_rng`, so the same request
yields byte-identical panels. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from fedcausal._constants import DEFAULT_ESTIMATION_WINDOW

if TYPE_CHECKING:
    import pandas as pd

    from fedcausal._typing import SurpriseLabel


@dataclass(frozen=True, slots=True)
class SyntheticPanel:
    """A synthetic event panel bundled with its ground-truth metadata.

    Attributes
    ----------
    returns:
        Wide panel of single-name simple returns (rows = trading date, columns =
        ticker).
    market:
        The 1-D market-factor return series used as the market-model regressor.
    announcement_dates:
        The FOMC announcement dates injected into the panel.
    surprises:
        The surprise label per announcement date (parallel to
        ``announcement_dates``).
    rate_sensitive:
        The subset of columns designated rate-sensitive (the DiD "treated" group
        that mechanically moves more around announcements).
    injected_car:
        The ground-truth cumulative abnormal return injected into rate-sensitive
        names over the event window (``0.0`` for a pure-noise panel).
    seed:
        The master seed the panel was generated from.
    """

    returns: pd.DataFrame
    market: pd.Series
    announcement_dates: list[date]
    surprises: list[SurpriseLabel]
    rate_sensitive: list[str]
    injected_car: float
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` (excludes bulk panels)."""
        return {
            "n_dates": int(self.returns.shape[0]),
            "n_names": int(self.returns.shape[1]),
            "n_events": len(self.announcement_dates),
            "rate_sensitive": list(self.rate_sensitive),
            "injected_car": float(self.injected_car),
            "seed": int(self.seed),
            "meta": dict(self.meta),
        }


def synthetic_event_panel(
    *,
    n_names: int = 30,
    n_events: int = 16,
    spacing: int = 40,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    event_half_width: int = 1,
    injected_car: float = 0.01,
    rate_sensitive_frac: float = 0.4,
    noise_scale: float = 0.01,
    seed: int = 7,
    start: date | None = None,
) -> SyntheticPanel:
    """Generate a synthetic event panel with a KNOWN injected CAR.

    Each name follows a one-factor market model (``alpha + beta * market + eps``).
    Around each FOMC announcement, a KNOWN ``injected_car`` is added to the event
    window of the rate-sensitive subset, split across the ``2k + 1`` event-window
    days, and a (smaller) sign-dependent move is added for hawkish/dovish events
    to create cross-sectional DiD heterogeneity. The estimation window carries NO
    injected effect, so an estimation-window-only market model is unbiased.

    Parameters
    ----------
    n_names:
        Number of single-name columns.
    n_events:
        Number of FOMC announcements to inject.
    spacing:
        Trading-day spacing between consecutive announcements (kept large enough
        that event windows never straddle).
    estimation_window:
        Pre-event estimation-window length (used to size the leading burn-in so
        the first event has enough history).
    event_half_width:
        The half-width ``k`` of the event window.
    injected_car:
        The ground-truth CAR injected into rate-sensitive names per event.
    rate_sensitive_frac:
        Fraction of names designated rate-sensitive (the DiD treated group).
    noise_scale:
        Idiosyncratic daily return standard deviation.
    seed:
        Master RNG seed.
    start:
        Optional first calendar date (defaults to a fixed reference date).

    Returns
    -------
    SyntheticPanel
        The panel plus its ground-truth metadata.

    Raises
    ------
    ValidationError
        If any size parameter is out of range.
    """
    raise NotImplementedError


def rate_sensitive_panel(
    *,
    seed: int = 7,
    injected_car: float = 0.012,
    **kwargs: Any,
) -> SyntheticPanel:
    """Generate a panel with pronounced rate-sensitivity DiD heterogeneity.

    A thin preset over :func:`synthetic_event_panel` that amplifies the gap
    between rate-sensitive ("treated") and insensitive ("control") names around
    hawkish vs. dovish surprises, so the difference-in-differences design has a
    clear (but still honestly fragile, net-of-cost) treated-minus-control signal.

    Parameters
    ----------
    seed:
        Master RNG seed.
    injected_car:
        The CAR injected into rate-sensitive names.
    **kwargs:
        Forwarded to :func:`synthetic_event_panel`.

    Returns
    -------
    SyntheticPanel
        The rate-sensitive panel with ground-truth metadata.
    """
    raise NotImplementedError


def pure_noise_panel(
    *,
    seed: int = 7,
    **kwargs: Any,
) -> SyntheticPanel:
    """Generate a NO-EFFECT panel (the honest-null control).

    Identical structure to :func:`synthetic_event_panel` but with
    ``injected_car=0`` and no sign-dependent move: there is genuinely no abnormal
    return around announcements. A correct pipeline must yield
    ``fed_effect_is_tradable=False`` and a placebo percentile of the observed CAR
    that is ~uniform on this panel.

    Parameters
    ----------
    seed:
        Master RNG seed.
    **kwargs:
        Forwarded to :func:`synthetic_event_panel` (``injected_car`` is forced
        to ``0.0``).

    Returns
    -------
    SyntheticPanel
        The pure-noise panel (``injected_car == 0.0``).
    """
    raise NotImplementedError
