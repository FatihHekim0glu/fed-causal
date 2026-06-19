"""Honest cross-sectional-heterogeneity framing (SCM demoted to descriptive).

This module wraps the DiD estimate (:mod:`fedcausal.did.model`) in the brief's
honest interpretation and computes the one quantity the verdict actually needs
from the DiD: the **net-of-cost tradable spread** implied by sorting names on
rate-sensitivity and going long/short around announcements.

The synthetic-control method (SCM) is DEMOTED to a descriptive diagnostic: it can
illustrate the treated-vs-synthetic-control gap, but it is NEVER presented as a
profit claim. The deliverable is the statement that FOMC moves are cross-sectional
HETEROGENEITY (rate-sensitive names move more, mechanically), not a placebo-robust
tradable alpha.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from fedcausal._constants import DEFAULT_COST_BPS

if TYPE_CHECKING:
    import numpy as np

    from fedcausal.did.model import DiDResult


@dataclass(frozen=True, slots=True)
class HeterogeneitySpread:
    """The descriptive heterogeneity spread and its net-of-cost tradability.

    Attributes
    ----------
    gross_spread:
        The gross treated-minus-control abnormal-return spread per announcement
        (the long/short rotation's average gross move).
    cost_bps:
        The round-trip transaction cost (basis points) charged against the spread.
    net_spread:
        ``gross_spread`` minus the per-rotation cost — the spread an actual
        long/short would capture net of costs.
    is_tradable_spread:
        ``True`` only if ``net_spread`` is positive AND statistically distinct
        from zero (this is one of FOUR independent conditions the overall verdict
        requires; it is FALSE by default for the honest-null deliverable).
    """

    gross_spread: float
    cost_bps: float
    net_spread: float
    is_tradable_spread: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the spread."""
        return asdict(self)


def heterogeneity_spread(
    did: DiDResult,
    spread_samples: np.ndarray,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    alpha: float = 0.05,
) -> HeterogeneitySpread:
    """Translate the DiD heterogeneity into a NET-OF-COST tradable spread test.

    Charges a round-trip ``cost_bps`` against the per-announcement treated-minus-
    control spread and flags it tradable ONLY if the net spread is both positive
    and statistically distinct from zero. By construction (mechanical
    heterogeneity, fragile after costs) this reads ``False`` on the honest-null
    deliverable.

    Parameters
    ----------
    did:
        The DiD interaction result (descriptive heterogeneity magnitude).
    spread_samples:
        Per-announcement treated-minus-control spread observations (for the
        net-of-cost significance test).
    cost_bps:
        Round-trip transaction cost in basis points.
    alpha:
        Significance level for the net-spread test.

    Returns
    -------
    HeterogeneitySpread
        The gross/net spread and its (default-``False``) tradability flag.

    Raises
    ------
    ValidationError
        If ``cost_bps`` is negative or ``spread_samples`` is empty.
    """
    raise NotImplementedError


def describe_heterogeneity(
    did: DiDResult,
    spread: HeterogeneitySpread,
) -> str:
    """Return the honest, non-promotional one-line interpretation of the DiD.

    Produces text framing the result as descriptive cross-sectional heterogeneity
    (rate-sensitive names move more, mechanically), explicitly NOT a tradable
    causal alpha — the language rendered in the README and the frontend caption.

    Parameters
    ----------
    did:
        The DiD interaction result.
    spread:
        The net-of-cost spread test.

    Returns
    -------
    str
        The honest interpretation string.
    """
    raise NotImplementedError
