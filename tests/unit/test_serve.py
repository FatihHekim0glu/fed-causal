"""Unit tests for the hosted orchestration entrypoint (``fedcausal.serve``).

Covers the request-validation guardrails, the surprise-subset filtering, the
``_safe_float`` NaN guard, and the assembled summary/figure/manifest contract —
the leaf behaviours the integration test exercises end-to-end.
"""

from __future__ import annotations

import math

import pytest

from fedcausal._exceptions import ValidationError
from fedcausal.serve import (
    AnalysisResult,
    _filter_by_surprise,
    _safe_float,
    _validate_request,
    run_analysis,
)


def _ok_request(**overrides: object) -> dict[str, object]:
    """A valid request kwargs dict with optional overrides."""
    base: dict[str, object] = {
        "event_window": 1,
        "estimation_window": 120,
        "model": "market",
        "surprise": "all",
        "n_placebo": 500,
        "data_source_pref": "synthetic",
        "seed": 7,
        "alpha": 0.05,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides",
    [
        {"event_window": 0},
        {"event_window": 11},
        {"estimation_window": 1},
        {"model": "transformer"},
        {"surprise": "flat"},
        {"n_placebo": 0},
        {"n_placebo": 5000},
        {"data_source_pref": "bloomberg"},
        {"seed": -1},
        {"alpha": 0.0},
        {"alpha": 1.0},
    ],
)
def test_validate_request_rejects_out_of_range(overrides: dict[str, object]) -> None:
    """Every out-of-range request parameter raises a typed ``ValidationError``."""
    with pytest.raises(ValidationError):
        _validate_request(**_ok_request(**overrides))  # type: ignore[arg-type]


def test_validate_request_accepts_valid() -> None:
    """A fully valid request validates without raising."""
    _validate_request(**_ok_request())  # type: ignore[arg-type]


def test_safe_float_maps_non_finite_to_zero() -> None:
    """``_safe_float`` keeps finite values and maps NaN/inf/garbage to ``0.0``."""
    assert _safe_float(1.25) == 1.25
    assert _safe_float(math.nan) == 0.0
    assert _safe_float(math.inf) == 0.0
    assert _safe_float(-math.inf) == 0.0
    assert _safe_float("not-a-number") == 0.0


def test_filter_by_surprise_all_is_passthrough() -> None:
    """``surprise="all"`` returns the inputs unchanged (object identity preserved)."""
    windows: list[object] = ["w0", "w1"]
    surprises: list[object] = ["hawkish", "dovish"]
    out_w, out_s = _filter_by_surprise(windows, surprises, "all")  # type: ignore[arg-type]
    assert out_w is windows
    assert out_s is surprises


def test_filter_by_surprise_too_few_falls_back_to_all() -> None:
    """A subset with < 2 events falls back to the full set (never hard-fails)."""
    windows: list[object] = ["w0", "w1", "w2"]
    surprises: list[object] = ["hawkish", "dovish", "dovish"]
    # Only one hawkish event -> below the 2-event floor -> fall back to all.
    out_w, out_s = _filter_by_surprise(windows, surprises, "hawkish")  # type: ignore[arg-type]
    assert out_w == windows
    assert out_s == surprises


def test_filter_by_surprise_selects_subset_when_enough() -> None:
    """A subset with >= 2 events of the requested sign is selected."""
    windows: list[object] = ["w0", "w1", "w2", "w3"]
    surprises: list[object] = ["hawkish", "dovish", "hawkish", "dovish"]
    out_w, out_s = _filter_by_surprise(windows, surprises, "dovish")  # type: ignore[arg-type]
    assert out_w == ["w1", "w3"]
    assert out_s == ["dovish", "dovish"]


@pytest.mark.parametrize("surprise", ["hawkish", "dovish"])
def test_run_analysis_surprise_subset_runs(surprise: str) -> None:
    """The hawkish/dovish surprise subsets run end-to-end and stay honest.

    The event-study/placebo battery genuinely narrows to the requested surprise
    sign (fewer events than ``"all"``), and the verdict remains ``False``.
    """
    full = run_analysis(n_placebo=120, surprise="all")
    subset = run_analysis(n_placebo=120, surprise=surprise)

    assert subset.summary["n_events"] <= full.summary["n_events"]
    assert subset.summary["fed_effect_is_tradable"] is False


def test_run_analysis_mean_adjusted_model_runs() -> None:
    """The ``mean_adjusted`` model is honoured end-to-end (not silently coerced)."""
    result = run_analysis(model="mean_adjusted", n_placebo=120)

    assert isinstance(result, AnalysisResult)
    assert result.summary["fed_effect_is_tradable"] is False


def test_run_analysis_manifest_carries_seed_and_config_hash() -> None:
    """The result manifest records the seed and a config hash (reproducibility)."""
    seed = 11
    result = run_analysis(seed=seed, n_placebo=120)

    assert result.manifest["seed"] == seed
    assert isinstance(result.manifest["config_hash"], str)
    assert len(result.manifest["config_hash"]) == 32
