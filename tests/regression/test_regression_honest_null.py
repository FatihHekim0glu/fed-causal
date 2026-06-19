"""Regression tests — known-CAR recovery, the honest-NULL guard, the reference.

The brief's regression plan, the deliverable's core claims pinned to behaviour:

- a known-CAR synthetic panel is recovered (CAR ≈ injected within tolerance);
- the honest-null guard: a pure-noise panel yields
  ``fed_effect_is_tradable=False`` after placebo + multiple testing, and the
  verdict is deterministic across ``PYTHONHASHSEED``;
- the placebo percentile of a no-effect panel is ~uniform (not systematically
  extreme);
- the committed deployed-default reference matches the live ``run_analysis``
  output (the backend serves a stable, reproducible result).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import cast

import numpy as np
import pandas as pd
import pytest

from fedcausal._constants import DEFAULT_ALPHA
from fedcausal.artifacts import load_reference
from fedcausal.data.synthetic import SyntheticPanel, pure_noise_panel
from fedcausal.events.windows import build_all_windows
from fedcausal.eventstudy.abnormal import cumulative_abnormal_returns, stack_event_cars
from fedcausal.eventstudy.placebo import placebo_distribution
from fedcausal.serve import _run_pipeline, run_analysis

pytestmark = pytest.mark.regression


def _grid(panel: SyntheticPanel) -> pd.DatetimeIndex:
    """The panel's trading-day index, typed as a ``DatetimeIndex``."""
    return cast("pd.DatetimeIndex", panel.returns.index)


def test_known_car_recovered_within_tolerance(
    synthetic_event_panel: SyntheticPanel,
) -> None:
    """The injected CAR is recovered from the synthetic panel within tolerance.

    The CAR is market-wide, so BOTH the rate-sensitive ("treated") and the control
    cross-sectional CAR recover the injected effect; the treated-minus-control gap
    is only the small hawkish/dovish tilt — the ground-truth recovery property
    plus the honest "no tradable cross-sectional gap" property.
    """
    panel = synthetic_event_panel
    grid = _grid(panel)
    windows = build_all_windows(grid, panel.announcement_dates, event_half_width=1)

    treated_cars: list[float] = []
    control_cars: list[float] = []
    for w in windows:
        result = cumulative_abnormal_returns(panel.returns, panel.market, w)
        treated_cars.append(float(result.car[panel.rate_sensitive].mean()))
        control_cars.append(float(result.car.drop(panel.rate_sensitive).mean()))

    treated_mean = float(np.mean(treated_cars))
    control_mean = float(np.mean(control_cars))
    assert treated_mean == pytest.approx(panel.injected_car, abs=panel.injected_car * 0.8)
    assert control_mean == pytest.approx(panel.injected_car, abs=panel.injected_car * 0.8)
    assert treated_mean > 0.0
    assert abs(treated_mean - control_mean) < panel.injected_car * 0.5


def test_pure_noise_panel_is_not_tradable() -> None:
    """Honest-null: a no-effect panel yields ``fed_effect_is_tradable=False``.

    On a genuinely effect-free panel the full pipeline (placebo + HAC + multiple
    testing + DiD) must clear NONE of the four verdict gates, so the headline
    boolean is unambiguously ``False``.
    """
    panel = pure_noise_panel(seed=7)
    outputs = _run_pipeline(
        panel,
        "synthetic",
        event_window=1,
        estimation_window=120,
        model="market",
        surprise="all",
        n_placebo=500,
        seed=7,
        alpha=DEFAULT_ALPHA,
    )
    summary = outputs.summary

    assert summary["fed_effect_is_tradable"] is False
    # The PRIMARY significance source (placebo) is non-significant on noise ...
    assert float(summary["placebo_pvalue"]) >= DEFAULT_ALPHA
    # ... and so is the HAC robustness gate.
    assert float(summary["car_hac_pvalue"]) >= DEFAULT_ALPHA


def test_pure_noise_verdict_deterministic_across_pythonhashseed() -> None:
    """The pure-noise verdict is invariant to ``PYTHONHASHSEED`` (seed-locked).

    The honest-null must not depend on the interpreter's hash randomization: two
    fresh subprocesses with different ``PYTHONHASHSEED`` values must produce the
    identical pure-noise verdict and key placebo/HAC scalars.
    """
    snippet = (
        "from fedcausal.data.synthetic import pure_noise_panel;"
        "from fedcausal.serve import _run_pipeline;"
        "p = pure_noise_panel(seed=7);"
        "o = _run_pipeline(p, 'synthetic', event_window=1, estimation_window=120,"
        " model='market', surprise='all', n_placebo=300, seed=7, alpha=0.05);"
        "s = o.summary;"
        "print(f\"{s['fed_effect_is_tradable']}|{s['placebo_pvalue']:.10f}|"
        "{s['car_hac_pvalue']:.10f}|{s['car_mean']:.10f}\")"
    )

    def _run(hashseed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=120,
        )
        return proc.stdout.strip()

    out_a = _run("0")
    out_b = _run("12345")
    assert out_a == out_b
    assert out_a.startswith("False|")


def test_placebo_percentile_centered_under_no_effect() -> None:
    """On a no-effect panel the observed CAR is NOT systematically extreme.

    Averaged over many independent pure-noise panels the observed mean-CAR placebo
    p-value is centred near 0.5 and the nominal-5% false-rejection rate is not
    inflated — a leaky or mis-scaled placebo null would push p-values to the tails.
    """
    p_values: list[float] = []
    for offset in range(30):
        panel = pure_noise_panel(seed=5000 + offset, n_events=12, n_names=24)
        grid = _grid(panel)
        windows = build_all_windows(grid, panel.announcement_dates, event_half_width=1)
        observed = float(stack_event_cars(panel.returns, panel.market, windows).mean())
        result = placebo_distribution(
            panel.returns,
            panel.market,
            panel.announcement_dates,
            observed,
            n_placebo=120,
            seed=29 + offset,
        )
        p_values.append(result.p_value)

    arr = np.asarray(p_values, dtype=np.float64)
    # A no-effect null is centred near 0.5; the band is deliberately generous so
    # the check rejects a systematically-extreme (leaky) null without being
    # brittle to the exact finite-sample seed range.
    assert 0.30 <= float(arr.mean()) <= 0.70
    assert float(np.mean(arr < 0.05)) <= 0.20


def test_reference_artifact_matches_live_run_analysis() -> None:
    """The committed reference matches the live ``run_analysis`` deployed default.

    The backend serves the committed ``reference.json`` for a stable result; this
    pins it to the live pipeline so the artifact can never silently drift.
    """
    reference = load_reference()
    request = reference["request"]
    live = run_analysis(
        event_window=int(request["event_window"]),
        estimation_window=int(request["estimation_window"]),
        model=str(request["model"]),
        surprise=str(request["surprise"]),
        n_placebo=int(request["n_placebo"]),
        data_source_pref=str(request["data_source_pref"]),
        seed=int(request["seed"]),
        alpha=float(request["alpha"]),
    )

    committed = reference["summary"]
    for key, value in committed.items():
        if isinstance(value, float):
            assert live.summary[key] == pytest.approx(value, rel=1e-9, abs=1e-12)
        else:
            assert live.summary[key] == value
    # The honest-null verdict is committed false in both the default and the
    # pure-noise control.
    assert committed["fed_effect_is_tradable"] is False
    assert reference["pure_noise_honest_null"]["fed_effect_is_tradable"] is False


def test_reference_known_car_recovery_is_honest() -> None:
    """The committed known-CAR recovery recovers the injected effect on treated names."""
    recovery = load_reference()["known_car_recovery"]
    injected = float(recovery["injected_car"])
    treated = float(recovery["recovered_treated_mean_car"])
    control = float(recovery["recovered_control_mean_car"])

    assert injected > 0.0
    # The CAR is market-wide: both treated and control recover it within tolerance;
    # the treated-minus-control gap is only the small tilt (not a tradable spread).
    assert treated == pytest.approx(injected, abs=injected * 0.8)
    assert control == pytest.approx(injected, abs=injected * 0.8)
    assert treated > 0.0
    assert abs(treated - control) < injected * 0.5
