#!/usr/bin/env python
"""Regenerate the committed deployed-default reference artifact.

Writes ``src/fedcausal/artifacts/reference.json`` from the seeded synthetic event
panel: the deployed-default summary, the known-CAR-recovery numbers, and the
pure-noise honest-null numbers. Fully deterministic (seed-locked), so re-running
it on a clean checkout reproduces the committed bytes.

Usage::

    python scripts/build_reference.py [--check]

``--check`` rebuilds the reference in memory and exits non-zero if it differs
from the committed file (used in CI to guard against drift) without rewriting it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fedcausal import __version__
from fedcausal._constants import DEFAULT_ALPHA, DEFAULT_ESTIMATION_GAP
from fedcausal.data.synthetic import pure_noise_panel, synthetic_event_panel
from fedcausal.events.windows import build_all_windows
from fedcausal.eventstudy.abnormal import cumulative_abnormal_returns
from fedcausal.serve import _run_pipeline, run_analysis

#: The committed artifact path (relative to the repository root).
_REFERENCE_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "fedcausal" / "artifacts" / "reference.json"
)

#: The deployed-default request the backend serves.
_DEFAULT_REQUEST: dict[str, Any] = {
    "event_window": 1,
    "estimation_window": 120,
    "model": "market",
    "surprise": "all",
    "n_placebo": 500,
    "data_source_pref": "synthetic",
    "seed": 7,
    "alpha": DEFAULT_ALPHA,
}


def _known_car_recovery() -> dict[str, Any]:
    """Recover the injected CAR on treated vs. control names (ground truth)."""
    panel = synthetic_event_panel(seed=7)
    windows = build_all_windows(
        panel.returns.index,  # type: ignore[arg-type]
        panel.announcement_dates,
        event_half_width=1,
        estimation_window=120,
        estimation_gap=DEFAULT_ESTIMATION_GAP,
    )
    treated: list[float] = []
    control: list[float] = []
    for w in windows:
        res = cumulative_abnormal_returns(panel.returns, panel.market, w)
        treated.append(float(res.car[panel.rate_sensitive].mean()))
        control.append(float(res.car.drop(panel.rate_sensitive).mean()))
    return {
        "injected_car": float(panel.injected_car),
        "recovered_treated_mean_car": float(np.mean(treated)),
        "recovered_control_mean_car": float(np.mean(control)),
        "n_events": len(windows),
    }


def _pure_noise_summary() -> dict[str, Any]:
    """The pure-noise honest-null summary (every verdict condition false)."""
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
    return outputs.summary


def build_reference() -> dict[str, Any]:
    """Build the full reference payload (deterministic, seed-locked)."""
    default = run_analysis(
        event_window=int(_DEFAULT_REQUEST["event_window"]),
        estimation_window=int(_DEFAULT_REQUEST["estimation_window"]),
        model=str(_DEFAULT_REQUEST["model"]),
        surprise=str(_DEFAULT_REQUEST["surprise"]),
        n_placebo=int(_DEFAULT_REQUEST["n_placebo"]),
        data_source_pref=str(_DEFAULT_REQUEST["data_source_pref"]),
        seed=int(_DEFAULT_REQUEST["seed"]),
        alpha=float(_DEFAULT_REQUEST["alpha"]),
    )
    return {
        "schema_version": 1,
        "package_version": __version__,
        "description": (
            "Committed deployed-default reference for the fed-causal hosted tool: "
            "the default summary the backend serves, plus the known-CAR recovery "
            "and the pure-noise honest-null numbers. Regenerate with "
            "scripts/build_reference.py."
        ),
        "request": dict(_DEFAULT_REQUEST),
        "summary": default.summary,
        "known_car_recovery": _known_car_recovery(),
        "pure_noise_honest_null": _pure_noise_summary(),
    }


def _serialize(reference: dict[str, Any]) -> str:
    """Canonical JSON serialization (sorted keys, strict, trailing newline)."""
    return json.dumps(reference, indent=2, allow_nan=False, sort_keys=True) + "\n"


def main() -> int:
    """Regenerate (or, with ``--check``, validate) the committed reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed reference differs from a fresh rebuild.",
    )
    args = parser.parse_args()

    payload = _serialize(build_reference())

    if args.check:
        existing = _REFERENCE_PATH.read_text(encoding="utf-8") if _REFERENCE_PATH.exists() else ""
        if existing != payload:
            print("reference.json is OUT OF DATE; run `python scripts/build_reference.py`.")
            return 1
        print("reference.json is up to date.")
        return 0

    _REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REFERENCE_PATH.write_text(payload, encoding="utf-8")
    print(f"wrote {_REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
