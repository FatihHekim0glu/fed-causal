"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by :func:`fedcausal._rng.make_rng` via the
synthetic-panel generators) and returns a :class:`fedcausal.SyntheticPanel`, so
tests across the suite share identical synthetic data with known structure:

- ``synthetic_event_panel`` — a one-factor market panel with a KNOWN injected CAR
  plus rate-sensitivity heterogeneity (the ground-truth recovery target).
- ``rate_sensitive_panel`` — an amplified treated-vs-control panel for the DiD.
- ``pure_noise`` — a no-effect panel (the honest-null control: a correct pipeline
  yields ``fed_effect_is_tradable=False`` and a ~uniform placebo percentile).

The fixtures call the synthetic generators lazily (only when a test requests
them), so collection succeeds while the generators are still stubs.

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fedcausal._rng import make_rng

if TYPE_CHECKING:
    from fedcausal.data.synthetic import SyntheticPanel

#: Master seed shared across the suite (a fixed date-like constant).
SEED: int = 20260619


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(SEED)


@pytest.fixture
def synthetic_event_panel() -> SyntheticPanel:
    """A synthetic event panel with a KNOWN injected CAR (ground truth).

    The event study must recover the injected CAR within tolerance and, after the
    placebo/HAC/multiple-testing gauntlet, must still report an honest verdict.
    """
    from fedcausal.data.synthetic import synthetic_event_panel as _make

    return _make(seed=SEED)


@pytest.fixture
def rate_sensitive_panel() -> SyntheticPanel:
    """A panel with pronounced rate-sensitivity DiD heterogeneity (treated vs control)."""
    from fedcausal.data.synthetic import rate_sensitive_panel as _make

    return _make(seed=SEED)


@pytest.fixture
def pure_noise() -> SyntheticPanel:
    """A NO-EFFECT panel — the honest-null control (``injected_car == 0``)."""
    from fedcausal.data.synthetic import pure_noise_panel as _make

    return _make(seed=SEED)
