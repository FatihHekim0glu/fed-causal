"""Truth-table tests for the PURE ``fed_effect_is_tradable`` verdict.

The verdict is ``True`` only when ALL FOUR conditions clear:
placebo-significant AND HAC-robust AND survives-multiple-testing AND a net-of-cost
tradable DiD spread. The table below pins every single-condition failure to
``False`` — including the HONEST-NULL row where a raw t-stat / HAC p-value is tiny
but the placebo percentile is insignificant (the move is NOT tradable).
"""

from __future__ import annotations

import itertools

import pytest

from fedcausal._constants import DEFAULT_ALPHA
from fedcausal._exceptions import ValidationError
from fedcausal.evaluation.verdict import (
    FedVerdict,
    VerdictInputs,
    derive_verdict,
    fed_effect_is_tradable,
)

pytestmark = pytest.mark.unit


def _inputs(
    *,
    placebo_pvalue: float = 0.01,
    hac_pvalue: float = 0.01,
    multiple_testing_survives: bool = True,
    did_net_spread: float = 0.02,
    did_spread_pvalue: float = 0.01,
) -> VerdictInputs:
    """All-clear inputs by default; override exactly one to break a condition."""
    return VerdictInputs(
        placebo_pvalue=placebo_pvalue,
        hac_pvalue=hac_pvalue,
        multiple_testing_survives=multiple_testing_survives,
        did_net_spread=did_net_spread,
        did_spread_pvalue=did_spread_pvalue,
    )


# --------------------------------------------------------------------------- #
# The only TRUE row                                                            #
# --------------------------------------------------------------------------- #
def test_all_conditions_clear_is_tradable() -> None:
    """The single ``True`` row: every condition clears."""
    inputs = _inputs()
    assert fed_effect_is_tradable(inputs) is True
    verdict = derive_verdict(inputs)
    assert verdict.fed_effect_is_tradable is True
    assert verdict.placebo_significant
    assert verdict.hac_robust
    assert verdict.survives_multiple_testing
    assert verdict.tradable_did_spread
    assert "Tradable" in verdict.rationale


# --------------------------------------------------------------------------- #
# Single-condition failures all read FALSE                                     #
# --------------------------------------------------------------------------- #
def test_placebo_insignificant_is_not_tradable() -> None:
    """HONEST-NULL: raw/HAC significant but placebo-insignificant -> NOT tradable."""
    inputs = _inputs(placebo_pvalue=0.40, hac_pvalue=0.0001)
    assert fed_effect_is_tradable(inputs) is False
    verdict = derive_verdict(inputs)
    assert verdict.fed_effect_is_tradable is False
    assert verdict.placebo_significant is False
    assert verdict.hac_robust is True  # raw/HAC would have "passed" on its own
    assert "placebo" in verdict.rationale


def test_hac_insignificant_is_not_tradable() -> None:
    inputs = _inputs(hac_pvalue=0.30)
    assert fed_effect_is_tradable(inputs) is False
    assert derive_verdict(inputs).hac_robust is False


def test_multiple_testing_failure_is_not_tradable() -> None:
    inputs = _inputs(multiple_testing_survives=False)
    assert fed_effect_is_tradable(inputs) is False
    assert derive_verdict(inputs).survives_multiple_testing is False


def test_negative_net_spread_is_not_tradable() -> None:
    """Costs eat the spread: a NEGATIVE net spread is never tradable."""
    inputs = _inputs(did_net_spread=-0.001, did_spread_pvalue=0.0001)
    assert fed_effect_is_tradable(inputs) is False
    assert derive_verdict(inputs).tradable_did_spread is False


def test_insignificant_spread_is_not_tradable() -> None:
    inputs = _inputs(did_net_spread=0.02, did_spread_pvalue=0.40)
    assert fed_effect_is_tradable(inputs) is False
    assert derive_verdict(inputs).tradable_did_spread is False


def test_zero_net_spread_is_not_tradable() -> None:
    """A net spread of exactly zero is not positive, so not tradable."""
    inputs = _inputs(did_net_spread=0.0, did_spread_pvalue=0.0001)
    assert fed_effect_is_tradable(inputs) is False


# --------------------------------------------------------------------------- #
# Exhaustive boolean truth table over the four conditions                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("placebo_ok", "hac_ok", "mt_ok", "did_ok"),
    list(itertools.product([True, False], repeat=4)),
)
def test_exhaustive_truth_table(placebo_ok: bool, hac_ok: bool, mt_ok: bool, did_ok: bool) -> None:
    """``True`` iff ALL four conditions clear; every other cell is ``False``."""
    inputs = VerdictInputs(
        placebo_pvalue=0.01 if placebo_ok else 0.40,
        hac_pvalue=0.01 if hac_ok else 0.40,
        multiple_testing_survives=mt_ok,
        did_net_spread=0.02 if did_ok else -0.02,
        did_spread_pvalue=0.01 if did_ok else 0.40,
    )
    expected = placebo_ok and hac_ok and mt_ok and did_ok
    assert fed_effect_is_tradable(inputs) is expected

    verdict = derive_verdict(inputs)
    assert verdict.fed_effect_is_tradable is expected
    assert verdict.placebo_significant is placebo_ok
    assert verdict.hac_robust is hac_ok
    assert verdict.survives_multiple_testing is mt_ok
    assert verdict.tradable_did_spread is did_ok


# --------------------------------------------------------------------------- #
# Boundary semantics: the p-value gate is STRICT ``< alpha``                   #
# --------------------------------------------------------------------------- #
def test_pvalue_equal_to_alpha_does_not_clear() -> None:
    """A p-value exactly at ``alpha`` is NOT significant (strict ``<``)."""
    inputs = _inputs(placebo_pvalue=DEFAULT_ALPHA)
    assert fed_effect_is_tradable(inputs) is False


def test_custom_alpha_changes_the_gate() -> None:
    """A looser ``alpha`` can flip a borderline case to tradable."""
    inputs = _inputs(
        placebo_pvalue=0.08,
        hac_pvalue=0.08,
        did_spread_pvalue=0.08,
    )
    assert fed_effect_is_tradable(inputs, alpha=0.05) is False
    assert fed_effect_is_tradable(inputs, alpha=0.10) is True


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_out_of_range_pvalue_raises(bad: float) -> None:
    with pytest.raises(ValidationError):
        fed_effect_is_tradable(_inputs(placebo_pvalue=bad))


def test_non_finite_net_spread_raises() -> None:
    with pytest.raises(ValidationError, match="did_net_spread"):
        fed_effect_is_tradable(_inputs(did_net_spread=float("nan")))


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.5, 2.0])
def test_bad_alpha_raises(bad_alpha: float) -> None:
    with pytest.raises(ValidationError):
        fed_effect_is_tradable(_inputs(), alpha=bad_alpha)


def test_verdict_is_frozen_and_serializable() -> None:
    verdict = derive_verdict(_inputs())
    assert isinstance(verdict, FedVerdict)
    payload = verdict.to_dict()
    assert payload["fed_effect_is_tradable"] is True
    assert set(payload) == {
        "fed_effect_is_tradable",
        "placebo_significant",
        "hac_robust",
        "survives_multiple_testing",
        "tradable_did_spread",
        "rationale",
    }
    with pytest.raises((AttributeError, TypeError)):
        verdict.fed_effect_is_tradable = False  # type: ignore[misc]


def test_verdict_inputs_to_dict_roundtrips() -> None:
    inputs = _inputs()
    payload = inputs.to_dict()
    assert payload["multiple_testing_survives"] is True
    assert VerdictInputs(**payload) == inputs
