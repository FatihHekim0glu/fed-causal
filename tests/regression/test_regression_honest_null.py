"""Regression tests — known-CAR recovery and the honest-NULL guard.

The brief's regression plan, the deliverable's core claims pinned to behaviour:

- a known-CAR synthetic panel is recovered (CAR ≈ injected within tolerance);
- the honest-null guard: a pure-noise panel yields
  ``fed_effect_is_tradable=False`` after placebo + multiple testing;
- the placebo percentile of a no-effect panel is ~uniform;
- golden CAR / DiD values are stable.

Authored sequentially once the compute kernels land — skipped in the scaffold.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

_SCAFFOLD_REASON = "scaffold: regression body authored once compute kernels land"


# NOTE: the authored bodies will request the seeded conftest fixtures
# ``synthetic_event_panel`` / ``pure_noise``; they are omitted from the scaffold
# signatures so the skip fires BEFORE fixture setup (which currently raises
# NotImplementedError from the stub generators).


def test_known_car_recovered_within_tolerance() -> None:
    """The injected CAR is recovered from the synthetic panel within tolerance."""
    pytest.skip(_SCAFFOLD_REASON)


def test_pure_noise_panel_is_not_tradable() -> None:
    """Honest-null: a no-effect panel yields ``fed_effect_is_tradable=False``."""
    pytest.skip(_SCAFFOLD_REASON)


def test_placebo_percentile_uniform_under_no_effect() -> None:
    """The placebo percentile of a no-effect panel is ~uniform."""
    pytest.skip(_SCAFFOLD_REASON)


def test_golden_car_and_did_values_stable() -> None:
    """Pinned golden CAR / DiD values do not drift."""
    pytest.skip(_SCAFFOLD_REASON)
