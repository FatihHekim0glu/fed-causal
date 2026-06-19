"""Scaffold smoke tests: committed data, constants, and public API surface.

These tests exercise the parts of the scaffold that are real on day one — the
committed FOMC calendar data, the numerical constants, the exception hierarchy,
and the curated public API — so the suite has a green floor while the compute
kernels are still stubs.
"""

from __future__ import annotations

import pytest

import fedcausal
from fedcausal._constants import (
    DEFAULT_ESTIMATION_GAP,
    DEFAULT_ESTIMATION_WINDOW,
    MAX_EVENT_HALF_WIDTH,
    MAX_PLACEBO_DRAWS,
)
from fedcausal._exceptions import (
    EventCalendarError,
    FedCausalError,
    InsufficientDataError,
    ValidationError,
    WindowOverlapError,
)
from fedcausal.events.calendar_data import (
    FOMC_ANNOUNCEMENTS,
    PRE_FIRST_TARGET_UPPER,
)


@pytest.mark.unit
def test_public_api_is_exported() -> None:
    """The curated ``__all__`` exposes the headline verdict and core entrypoints."""
    for name in (
        "fed_effect_is_tradable",
        "derive_verdict",
        "run_analysis",
        "synthetic_event_panel",
        "pure_noise_panel",
        "load_fomc_calendar",
        "newey_west_se",
    ):
        assert name in fedcausal.__all__
        assert hasattr(fedcausal, name)
    assert fedcausal.__version__ == "0.1.0"


@pytest.mark.unit
def test_constants_are_sane() -> None:
    """The window/cap constants satisfy the leakage geometry preconditions."""
    assert DEFAULT_ESTIMATION_GAP >= 1  # strictly positive gap => no overlap
    assert DEFAULT_ESTIMATION_WINDOW > MAX_EVENT_HALF_WIDTH
    assert MAX_EVENT_HALF_WIDTH >= 1
    assert MAX_PLACEBO_DRAWS >= 500


@pytest.mark.unit
def test_exception_hierarchy() -> None:
    """All library errors descend from ``FedCausalError``; subclassing is correct."""
    assert issubclass(ValidationError, FedCausalError)
    assert issubclass(InsufficientDataError, ValidationError)
    assert issubclass(WindowOverlapError, ValidationError)
    assert issubclass(EventCalendarError, FedCausalError)


@pytest.mark.unit
def test_committed_fomc_calendar_is_well_formed() -> None:
    """The committed FOMC calendar is non-empty, chronological, and plausibly valued."""
    assert len(FOMC_ANNOUNCEMENTS) >= 8  # at least one year of meetings
    dates = [d for d, _ in FOMC_ANNOUNCEMENTS]
    assert dates == sorted(dates), "FOMC announcement dates must be chronological"
    assert len(set(dates)) == len(dates), "FOMC announcement dates must be unique"
    for _date, upper in FOMC_ANNOUNCEMENTS:
        assert 0.0 <= upper <= 12.0, "target-rate upper bound out of plausible range"
    assert 0.0 <= PRE_FIRST_TARGET_UPPER <= 12.0


@pytest.mark.unit
def test_committed_calendar_has_hawkish_dovish_and_neutral_moves() -> None:
    """The committed snapshot spans hikes, cuts, and holds (so all surprise signs exist)."""
    from itertools import pairwise

    uppers = [PRE_FIRST_TARGET_UPPER] + [u for _, u in FOMC_ANNOUNCEMENTS]
    deltas = [b - a for a, b in pairwise(uppers)]
    assert any(d > 0 for d in deltas), "expected at least one hike (hawkish)"
    assert any(d < 0 for d in deltas), "expected at least one cut (dovish)"
    assert any(d == 0 for d in deltas), "expected at least one hold (neutral)"
