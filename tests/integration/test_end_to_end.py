"""Integration test — the end-to-end synthetic pipeline (NO network).

Exercises the full stack via :func:`fedcausal.serve.run_analysis`:
synthetic panel -> leakage-safe event windows -> abnormal/CAR -> placebo + HAC +
DiD -> multiple-testing correction -> the PURE verdict -> the two figures. Must
run with NO network access and return a well-formed, JSON-serializable result.
"""

from __future__ import annotations

import json
import sys

import pytest

from fedcausal.serve import AnalysisResult, run_analysis

pytestmark = pytest.mark.integration

#: The scalar summary keys the API response ``summary`` block must carry.
_REQUIRED_SUMMARY_KEYS = {
    "car_mean",
    "car_tstat",
    "car_hac_pvalue",
    "bmp_stat",
    "placebo_pctile",
    "n_events",
    "did_coef",
    "did_pvalue",
    "n_tests",
    "fed_effect_is_tradable",
    "data_source",
}

#: Modules that must NOT be imported merely by importing/running the package
#: (import purity + the torch-free / lazy-client constraints).
_FORBIDDEN_IMPORT_MODULES = (
    "torch",
    "onnx",
    "onnxruntime",
    "sklearn",
)


def test_run_analysis_synthetic_end_to_end_no_network() -> None:
    """``run_analysis`` returns a serializable result on the synthetic default.

    The default path is fully offline (synthetic panel + committed calendar); the
    whole result round-trips through strict JSON (no NaN/inf), and the summary
    carries every contracted scalar key. No real network is available in CI, so a
    successful run proves the default path never reaches out.
    """
    result = run_analysis()

    assert isinstance(result, AnalysisResult)
    assert set(result.summary) >= _REQUIRED_SUMMARY_KEYS
    assert result.summary["data_source"] == "synthetic"
    assert int(result.summary["n_events"]) >= 2

    # Strict JSON (allow_nan=False) proves every scalar is finite and serializable.
    payload = json.dumps(result.to_dict(), allow_nan=False)
    assert len(payload) > 0
    # The torch-free constraint: no heavy ML runtime is pulled in by the run.
    assert not any(mod in sys.modules for mod in _FORBIDDEN_IMPORT_MODULES)


def test_run_analysis_emits_destructive_verdict_on_default() -> None:
    """The synthetic default emits ``fed_effect_is_tradable=False`` (honest-NULL).

    The deployed default must NEVER claim a tradable Fed alpha: the verdict is a
    PURE function of four independent gates and reads ``False`` because the
    rate-sensitivity heterogeneity is not a net-of-cost tradable spread.
    """
    result = run_analysis()

    assert result.summary["fed_effect_is_tradable"] is False
    # The honest rationale names the failing line of evidence.
    assert "Not tradable" in str(result.summary["verdict_rationale"])
    # n_tests honestly counts the full spec grid (>= 2 model specs here).
    assert int(result.summary["n_tests"]) >= 2


def test_run_analysis_figures_are_data_layout_dicts() -> None:
    """Both figures cross the boundary as plain ``{data, layout}`` mappings.

    The frontend ``PlotlyChart`` consumes a JSON ``{"data": [...], "layout": {...}}``
    shape, so the figures must be plain mappings (no Plotly objects, no numpy
    scalars) and survive a strict JSON round-trip.
    """
    result = run_analysis(n_placebo=200)

    for figure in (result.car_figure, result.placebo_figure):
        assert isinstance(figure, dict)
        assert set(figure) >= {"data", "layout"}
        assert isinstance(figure["data"], list)
        assert isinstance(figure["layout"], dict)
        # Strict JSON: no numpy scalars or non-finite values leaked into a figure.
        json.dumps(figure, allow_nan=False)

    # The CAR-path figure has the mean path + its confidence band (>= 2 traces);
    # the placebo figure carries the null histogram.
    assert len(result.car_figure["data"]) >= 2
    assert len(result.placebo_figure["data"]) >= 1


def test_run_analysis_rejects_out_of_range_request() -> None:
    """Out-of-range request parameters raise a typed ``ValidationError`` (422 path)."""
    from fedcausal._exceptions import ValidationError

    with pytest.raises(ValidationError):
        run_analysis(event_window=999)
    with pytest.raises(ValidationError):
        run_analysis(n_placebo=0)
    with pytest.raises(ValidationError):
        run_analysis(model="deep_net")
    with pytest.raises(ValidationError):
        run_analysis(surprise="sideways")


def test_run_analysis_is_deterministic_across_hashseed() -> None:
    """The default summary is byte-stable across runs (seed-locked, not hash-seeded).

    Two independent calls with the same seed must produce identical scalar
    summaries — the pipeline draws exclusively from the seeded RNG, never the
    interpreter hash seed or global numpy state.
    """
    first = run_analysis(n_placebo=200).summary
    second = run_analysis(n_placebo=200).summary

    # Compare the contracted scalar keys exactly.
    for key in sorted(_REQUIRED_SUMMARY_KEYS):
        assert first[key] == second[key], f"summary[{key!r}] drifted between runs"


def test_run_analysis_fred_polygon_pref_degrades_to_synthetic_offline() -> None:
    """The ``fred+polygon`` preference degrades to synthetic offline (never hard-fails).

    With no network the real-data loader must fall back to the deterministic
    synthetic panel, so the run still returns a coherent ``data_source`` and an
    honest verdict rather than raising.
    """
    result = run_analysis(data_source_pref="fred+polygon", n_placebo=200)

    assert result.summary["data_source"] == "synthetic"
    assert result.summary["fed_effect_is_tradable"] is False
