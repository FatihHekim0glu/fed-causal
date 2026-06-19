"""Integration test — the end-to-end synthetic pipeline (NO network).

Exercises the full stack via :func:`fedcausal.serve.run_analysis`:
synthetic panel -> leakage-safe event windows -> abnormal/CAR -> placebo + HAC +
DiD -> multiple-testing correction -> the PURE verdict -> the two figures. Must
run with NO network access and return a well-formed, JSON-serializable result.

Authored sequentially once the compute kernels land — skipped in the scaffold.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

_SCAFFOLD_REASON = "scaffold: integration body authored once compute kernels land"


def test_run_analysis_synthetic_end_to_end_no_network() -> None:
    """``run_analysis`` returns a serializable result on the synthetic default."""
    pytest.skip(_SCAFFOLD_REASON)


def test_run_analysis_emits_destructive_verdict_on_default() -> None:
    """The synthetic default emits ``fed_effect_is_tradable=False`` (honest-NULL)."""
    pytest.skip(_SCAFFOLD_REASON)


def test_run_analysis_figures_are_data_layout_dicts() -> None:
    """Both figures cross the boundary as plain ``{data, layout}`` mappings."""
    pytest.skip(_SCAFFOLD_REASON)
