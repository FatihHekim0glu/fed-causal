"""Single orchestration entrypoint for the hosted tool (``run_analysis``).

The FastAPI router (``api/routers/fed_causal.py``) calls exactly one function —
:func:`run_analysis` — which loads the event panel (synthetic by default), runs
the full leakage-free stack (estimation-window-only abnormal returns -> CAR ->
BMP/HAC tests -> placebo-date null -> rate-sensitivity DiD -> multiple-testing
correction -> the PURE verdict), and returns a plain, JSON-serializable summary
plus the two figures.

It NEVER refits a heavy model and NEVER hard-fails on a data-provider error: the
real-data path degrades to the synthetic/committed panel. Importing this module
has no side effects (no network, no statsmodels at import).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fedcausal._constants import (
    DEFAULT_ALPHA,
    DEFAULT_ESTIMATION_WINDOW,
)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """The full, JSON-serializable result of one hosted analysis run.

    Attributes
    ----------
    summary:
        Scalar summary fields mirroring the API response ``summary`` block
        (``car_mean``, ``car_tstat``, ``car_hac_pvalue``, ``bmp_stat``,
        ``placebo_pctile``, ``n_events``, ``did_coef``, ``did_pvalue``,
        ``n_tests``, ``fed_effect_is_tradable``, ``data_source``).
    car_figure:
        The CAR-path ``{"data", "layout"}`` figure (with CI band).
    placebo_figure:
        The placebo null-distribution ``{"data", "layout"}`` figure with the
        observed CAR marked.
    manifest:
        The reproducibility manifest (git sha, config hash, seed).
    """

    summary: dict[str, Any]
    car_figure: dict[str, Any]
    placebo_figure: dict[str, Any]
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the whole result."""
        return asdict(self)


def run_analysis(
    *,
    event_window: int = 1,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    model: str = "market",
    surprise: str = "all",
    n_placebo: int = 500,
    data_source_pref: str = "synthetic",
    seed: int = 7,
    alpha: float = DEFAULT_ALPHA,
) -> AnalysisResult:
    """Run the end-to-end fed-causal analysis and return a serializable result.

    Pipeline: load the event panel (synthetic default; ``fred+polygon`` degrades
    to synthetic on failure) -> build leakage-safe windows -> fit the
    estimation-window-only market model -> abnormal returns + CAR -> cross-
    sectional t / BMP / HAC tests -> placebo-date null (PRIMARY significance) ->
    rate-sensitivity DiD with clustered SEs -> Benjamini-Hochberg / Romano-Wolf
    correction over the full spec grid -> the PURE ``fed_effect_is_tradable``
    verdict -> the CAR-path and placebo figures.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k`` (window ``[-k, +k]``; capped).
    estimation_window:
        The pre-event estimation-window length.
    model:
        The expected-return model (``"market"`` or ``"mean_adjusted"``).
    surprise:
        The surprise subset (``"all"``, ``"hawkish"`` or ``"dovish"``).
    n_placebo:
        Number of placebo-date draws (capped).
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.
    alpha:
        Significance level applied to every gate and the verdict.

    Returns
    -------
    AnalysisResult
        The summary scalars, the two figures, and the run manifest.

    Raises
    ------
    ValidationError
        If any request parameter is out of range. Data-provider failures DO NOT
        raise: they degrade to the synthetic/committed panel.
    """
    raise NotImplementedError
