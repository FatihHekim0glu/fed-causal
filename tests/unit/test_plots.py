"""Unit tests for the lazy Plotly figure builders (:mod:`fedcausal.plots`).

Each builder must:

- return a plain ``{"data", "layout"}`` mapping that survives ``json.dumps`` (no
  numpy scalars / Plotly objects leak across the API boundary);
- render the intended marks (CAR path + CI band + zero line; placebo histogram +
  observed-CAR marker; DiD point + error bar + zero line); and
- raise :class:`fedcausal.ValidationError` on malformed input.

Importing :mod:`fedcausal.plots` must NOT import Plotly (it is lazily imported
inside the builders); this is asserted here too.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fedcausal._exceptions import ValidationError
from fedcausal.plots import (
    car_path_figure,
    did_coefficient_figure,
    placebo_histogram_figure,
)

pytestmark = pytest.mark.unit


def _assert_json_serializable(fig: dict[str, object]) -> None:
    """Assert the figure is a ``{data, layout}`` mapping that ``json.dumps`` accepts."""
    assert set(fig.keys()) >= {"data", "layout"}
    assert isinstance(fig["data"], list)
    assert isinstance(fig["layout"], dict)
    # The decisive contract: no numpy/Plotly object leaks across the boundary.
    json.dumps(fig)


# --------------------------------------------------------------------------- #
# car_path_figure                                                             #
# --------------------------------------------------------------------------- #
def test_car_path_figure_shape_and_traces() -> None:
    """The CAR path figure renders the band edges, the path, and a zero line."""
    car = np.array([0.001, 0.0025, 0.002], dtype=np.float64)
    lower = np.array([-0.001, -0.0005, -0.001], dtype=np.float64)
    upper = np.array([0.003, 0.005, 0.004], dtype=np.float64)

    fig = car_path_figure(car, lower, upper, event_half_width=1)

    _assert_json_serializable(fig)
    # Two band-edge traces + the mean-CAR path trace.
    assert len(fig["data"]) == 3
    # x-axis runs -k..+k.
    path_trace = fig["data"][-1]
    assert path_trace["x"] == [-1, 0, 1]
    assert path_trace["y"] == pytest.approx(car.tolist())
    # The zero reference line is present.
    assert fig["layout"]["shapes"]


def test_car_path_figure_larger_window() -> None:
    """A k=2 window yields a 5-day x-axis ``-2..+2``."""
    car = np.linspace(0.0, 0.004, 5).astype(np.float64)
    band = np.full(5, 0.002, dtype=np.float64)
    fig = car_path_figure(car, car - band, car + band, event_half_width=2)
    assert fig["data"][-1]["x"] == [-2, -1, 0, 1, 2]


@pytest.mark.parametrize(
    ("car", "lower", "upper", "k"),
    [
        # NaN in the path.
        (np.array([np.nan, 1.0, 2.0]), np.zeros(3), np.ones(3), 1),
        # Mismatched lengths.
        (np.array([1.0, 2.0, 3.0]), np.zeros(2), np.ones(3), 1),
        # Length inconsistent with the half-width (not 2k+1).
        (np.array([1.0, 2.0]), np.zeros(2), np.ones(2), 1),
        # lower > upper somewhere.
        (np.array([1.0, 2.0, 3.0]), np.full(3, 2.0), np.ones(3), 1),
        # Non-positive half-width.
        (np.array([1.0]), np.zeros(1), np.ones(1), 0),
        # Empty.
        (np.empty(0), np.empty(0), np.empty(0), 1),
    ],
)
def test_car_path_figure_rejects_bad_input(
    car: np.ndarray, lower: np.ndarray, upper: np.ndarray, k: int
) -> None:
    """Malformed CAR-path inputs raise ``ValidationError``."""
    with pytest.raises(ValidationError):
        car_path_figure(car, lower, upper, event_half_width=k)


# --------------------------------------------------------------------------- #
# placebo_histogram_figure                                                    #
# --------------------------------------------------------------------------- #
def test_placebo_histogram_figure_marks_observed() -> None:
    """The placebo figure is a histogram with a vertical observed-CAR marker."""
    draws = np.random.default_rng(0).normal(0.0, 0.001, size=256).astype(np.float64)
    observed = 0.0007

    fig = placebo_histogram_figure(draws, observed, percentile=64.0)

    _assert_json_serializable(fig)
    assert len(fig["data"]) == 1
    assert fig["data"][0]["type"] == "histogram"
    # The observed-CAR vertical marker sits at x == observed.
    shapes = fig["layout"]["shapes"]
    assert shapes and shapes[0]["x0"] == pytest.approx(observed)
    # The percentile is surfaced in the annotation text.
    annotations = fig["layout"]["annotations"]
    assert "64.0" in annotations[0]["text"]


def test_placebo_histogram_figure_without_percentile() -> None:
    """Omitting the percentile still produces a valid, marked histogram."""
    draws = np.array([0.0, 0.001, -0.001, 0.0005], dtype=np.float64)
    fig = placebo_histogram_figure(draws, 0.0)
    _assert_json_serializable(fig)
    assert fig["layout"]["annotations"][0]["text"] == "observed CAR"


@pytest.mark.parametrize(
    ("draws", "observed", "percentile"),
    [
        (np.array([np.inf, 1.0]), 0.0, None),  # non-finite draw
        (np.empty(0), 0.0, None),  # empty draws
        (np.array([1.0, 2.0]), float("nan"), None),  # non-finite observed
        (np.array([1.0, 2.0]), 0.5, -1.0),  # percentile < 0
        (np.array([1.0, 2.0]), 0.5, 101.0),  # percentile > 100
    ],
)
def test_placebo_histogram_figure_rejects_bad_input(
    draws: np.ndarray, observed: float, percentile: float | None
) -> None:
    """Malformed placebo inputs raise ``ValidationError``."""
    with pytest.raises(ValidationError):
        placebo_histogram_figure(draws, observed, percentile=percentile)


# --------------------------------------------------------------------------- #
# did_coefficient_figure                                                      #
# --------------------------------------------------------------------------- #
def test_did_coefficient_figure_point_and_error_bar() -> None:
    """The DiD figure renders the point estimate, its CI error bar, and a zero line."""
    fig = did_coefficient_figure(0.01, -0.002, 0.022)

    _assert_json_serializable(fig)
    assert len(fig["data"]) == 1
    trace = fig["data"][0]
    assert trace["y"] == pytest.approx([0.01])
    # Asymmetric error bar = (upper - coef, coef - lower).
    assert trace["error_y"]["array"] == pytest.approx([0.012])
    assert trace["error_y"]["arrayminus"] == pytest.approx([0.012])
    # The zero reference line is present.
    assert fig["layout"]["shapes"]


@pytest.mark.parametrize(
    ("coef", "lower", "upper"),
    [
        (0.01, 0.05, 0.02),  # lower > upper
        (float("inf"), -1.0, 1.0),  # non-finite coef
        (0.0, float("nan"), 1.0),  # non-finite lower
        (0.0, -1.0, float("inf")),  # non-finite upper
    ],
)
def test_did_coefficient_figure_rejects_bad_input(coef: float, lower: float, upper: float) -> None:
    """Malformed DiD-coefficient inputs raise ``ValidationError``."""
    with pytest.raises(ValidationError):
        did_coefficient_figure(coef, lower, upper)


# --------------------------------------------------------------------------- #
# laziness                                                                     #
# --------------------------------------------------------------------------- #
def test_importing_plots_does_not_import_plotly() -> None:
    """Importing :mod:`fedcausal.plots` in a fresh interpreter pulls in no Plotly."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import fedcausal.plots\n"
        "assert 'plotly' not in sys.modules, 'plotly leaked on import'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
