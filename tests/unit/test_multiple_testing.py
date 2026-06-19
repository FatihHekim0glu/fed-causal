"""Unit tests for the multiple-testing corrections (validation + serialization)."""

from __future__ import annotations

import numpy as np
import pytest

from fedcausal._exceptions import ValidationError
from fedcausal._rng import make_rng
from fedcausal.evaluation.multiple_testing import (
    MultipleTestingResult,
    benjamini_hochberg,
    romano_wolf,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg validation                                               #
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_rejects_empty_input() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        benjamini_hochberg(np.array([], dtype=float))


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan, np.inf])
def test_benjamini_hochberg_rejects_out_of_range_pvalues(bad: float) -> None:
    with pytest.raises(ValidationError):
        benjamini_hochberg(np.array([0.1, bad]))


def test_benjamini_hochberg_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValidationError, match="1-dimensional"):
        benjamini_hochberg(np.array([[0.1, 0.2]]))


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_benjamini_hochberg_rejects_bad_alpha(bad_alpha: float) -> None:
    with pytest.raises(ValidationError):
        benjamini_hochberg(np.array([0.01, 0.02]), alpha=bad_alpha)


def test_benjamini_hochberg_all_null_no_survivors() -> None:
    result = benjamini_hochberg(np.array([0.6, 0.7, 0.8, 0.9]), alpha=0.05)
    assert not result.any_survives
    assert not result.rejected.any()


def test_benjamini_hochberg_to_dict_is_json_plain() -> None:
    result = benjamini_hochberg(np.array([0.001, 0.5]), alpha=0.05)
    payload = result.to_dict()
    assert payload["method"] == "benjamini_hochberg"
    assert payload["n_tests"] == 2
    assert isinstance(payload["adjusted_pvalues"], list)
    assert all(isinstance(x, float) for x in payload["adjusted_pvalues"])
    assert isinstance(payload["rejected"], list)
    assert all(isinstance(x, bool) for x in payload["rejected"])
    assert isinstance(payload["any_survives"], bool)


def test_benjamini_hochberg_result_is_frozen() -> None:
    result = benjamini_hochberg(np.array([0.01, 0.2]))
    assert isinstance(result, MultipleTestingResult)
    with pytest.raises((AttributeError, TypeError)):
        result.method = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Romano-Wolf validation                                                       #
# --------------------------------------------------------------------------- #
def test_romano_wolf_rejects_empty_statistics() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        romano_wolf(np.array([], dtype=float), np.zeros((0, 3)))


def test_romano_wolf_rejects_misshaped_null() -> None:
    with pytest.raises(ValidationError, match="2-dimensional"):
        romano_wolf(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_romano_wolf_rejects_two_dimensional_statistics() -> None:
    with pytest.raises(ValidationError, match="1-dimensional"):
        romano_wolf(np.array([[1.0, 2.0]]), np.zeros((2, 50)))


def test_romano_wolf_rejects_row_mismatch() -> None:
    with pytest.raises(ValidationError, match="one row per spec"):
        romano_wolf(np.array([1.0, 2.0, 3.0]), np.zeros((2, 50)))


def test_romano_wolf_rejects_empty_resample_columns() -> None:
    with pytest.raises(ValidationError, match="at least one resample column"):
        romano_wolf(np.array([1.0, 2.0]), np.zeros((2, 0)))


def test_romano_wolf_rejects_non_finite() -> None:
    rng = make_rng(1)
    null = rng.standard_normal((2, 100))
    with pytest.raises(ValidationError, match="finite"):
        romano_wolf(np.array([np.inf, 1.0]), null)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.2])
def test_romano_wolf_rejects_bad_alpha(bad_alpha: float) -> None:
    rng = make_rng(1)
    null = rng.standard_normal((2, 100))
    with pytest.raises(ValidationError):
        romano_wolf(np.array([1.0, 2.0]), null, alpha=bad_alpha)


def test_romano_wolf_adjusted_pvalues_in_unit_interval() -> None:
    rng = make_rng(3)
    null = rng.standard_normal((4, 500))
    statistics = np.array([0.1, 5.0, -3.0, 0.0])
    result = romano_wolf(statistics, null, alpha=0.05)
    assert np.all(result.adjusted_pvalues >= 0.0)
    assert np.all(result.adjusted_pvalues <= 1.0)
    assert result.n_tests == 4


def test_romano_wolf_stepdown_monotone_in_extremity_order() -> None:
    """Stepdown p-values are non-decreasing from most to least extreme spec."""
    rng = make_rng(5)
    null = rng.standard_normal((6, 800))
    statistics = np.array([6.0, 4.0, 2.0, 1.0, 0.5, 0.1])  # already descending |stat|
    result = romano_wolf(statistics, null, alpha=0.05)
    order = np.argsort(np.abs(statistics))[::-1]
    ordered = result.adjusted_pvalues[order]
    assert np.all(np.diff(ordered) >= -1e-12)


def test_romano_wolf_to_dict_is_json_plain() -> None:
    rng = make_rng(9)
    null = rng.standard_normal((3, 200))
    result = romano_wolf(np.array([5.0, 0.1, 0.2]), null, alpha=0.05)
    payload = result.to_dict()
    assert payload["method"] == "romano_wolf"
    assert payload["n_tests"] == 3
    assert all(isinstance(x, float) for x in payload["adjusted_pvalues"])
    assert all(isinstance(x, bool) for x in payload["rejected"])
