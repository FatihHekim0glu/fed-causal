"""Unit tests for the difference-in-differences group (``did``).

Covers the two group files:

- ``model.py``: ``build_did_panel`` (long-format DiD panel from per-event abnormal
  matrices, filtered to the hawkish/dovish contrast) and ``estimate_did`` (the
  clustered-SE OLS interaction estimate), including the DiD coefficient recovering
  an injected heterogeneity on ``rate_sensitive_panel`` and the degenerate-input
  guards;
- ``heterogeneity.py``: ``heterogeneity_spread`` (the net-of-cost tradable-spread
  test, FALSE by default) and ``describe_heterogeneity`` (the honest, non-
  promotional interpretation string).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from fedcausal._constants import DEFAULT_COST_BPS
from fedcausal._exceptions import InsufficientDataError, ValidationError
from fedcausal.data.synthetic import SyntheticPanel
from fedcausal.did.heterogeneity import (
    HeterogeneitySpread,
    describe_heterogeneity,
    heterogeneity_spread,
)
from fedcausal.did.model import (
    DiDResult,
    build_did_panel,
    estimate_did,
)
from fedcausal.events.windows import EventWindows, build_all_windows
from fedcausal.eventstudy.abnormal import cumulative_abnormal_returns

# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _grid(panel: SyntheticPanel) -> pd.DatetimeIndex:
    """The panel's trading-day index, typed as a ``DatetimeIndex``."""
    return cast("pd.DatetimeIndex", panel.returns.index)


def _windows(panel: SyntheticPanel, k: int = 1) -> list[EventWindows]:
    return build_all_windows(_grid(panel), panel.announcement_dates, event_half_width=k)


def _abnormal_by_event(panel: SyntheticPanel, windows: list[EventWindows]) -> list[pd.DataFrame]:
    return [cumulative_abnormal_returns(panel.returns, panel.market, w).abnormal for w in windows]


def _surprises_for(panel: SyntheticPanel, windows: list[EventWindows]) -> list:
    """Align the panel's surprise labels to the (sorted/filtered) built windows."""
    return [panel.surprises[panel.announcement_dates.index(w.announcement_date)] for w in windows]


def _event_spreads(panel: SyntheticPanel, abnormal_by_event: list[pd.DataFrame]) -> np.ndarray:
    """Per-event treated-mean-minus-control-mean CAR spread."""
    spreads: list[float] = []
    for abnormal in abnormal_by_event:
        car = abnormal.sum(axis=0)
        treated = float(car[panel.rate_sensitive].mean())
        control = float(car.drop(panel.rate_sensitive).mean())
        spreads.append(treated - control)
    return np.asarray(spreads, dtype=np.float64)


def _did_result(coef: float = 0.01, t_stat: float = 3.0, cluster: str = "event") -> DiDResult:
    return DiDResult(
        coef=coef,
        std_error=abs(coef / t_stat) if t_stat else 0.01,
        t_stat=t_stat,
        p_value=0.01,
        n_obs=200,
        n_clusters=12,
        cluster=cluster,
    )


# --------------------------------------------------------------------------- #
# model.py — build_did_panel                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_build_did_panel_columns_and_dtypes(rate_sensitive_panel: SyntheticPanel) -> None:
    """The panel has exactly the fixed DiD columns and the right encodings."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    assert list(dp.columns) == ["y", "treated", "post", "interaction", "event", "name"]
    # interaction == treated * post for every row.
    assert (dp["interaction"] == dp["treated"] * dp["post"]).all()
    # treated flags exactly the rate-sensitive names.
    treated_names = set(dp.loc[dp["treated"] == 1, "name"].astype(str))
    assert treated_names == set(panel.rate_sensitive)


@pytest.mark.unit
def test_build_did_panel_drops_neutral_events(rate_sensitive_panel: SyntheticPanel) -> None:
    """Neutral (no-change) events carry no post contrast and are dropped."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    surprises = _surprises_for(panel, windows)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, surprises, panel.rate_sensitive)
    n_hawkish = sum(s == "hawkish" for s in surprises)
    n_dovish = sum(s == "dovish" for s in surprises)
    n_kept_events = int(dp["event"].nunique())
    assert n_kept_events == n_hawkish + n_dovish
    # post == 1 exactly for the hawkish events, 0 for dovish.
    assert set(dp.loc[dp["post"] == 1, "event"]).isdisjoint(dp.loc[dp["post"] == 0, "event"])


@pytest.mark.unit
def test_build_did_panel_post_maps_treated_surprise(rate_sensitive_panel: SyntheticPanel) -> None:
    """``post`` is 1 for the treated surprise and 0 for the control surprise."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    surprises = _surprises_for(panel, windows)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(
        abn, surprises, panel.rate_sensitive, treated_surprise="dovish", control_surprise="hawkish"
    )
    # Now dovish events are post==1; confirm the mapping flipped vs the default.
    dovish_events = {i for i, s in enumerate(surprises) if s == "dovish"}
    post_events = set(dp.loc[dp["post"] == 1, "event"].astype(int))
    assert post_events == dovish_events


@pytest.mark.unit
def test_build_did_panel_empty_input_raises() -> None:
    """An empty ``abnormal_by_event`` raises ``ValidationError``."""
    with pytest.raises(ValidationError):
        build_did_panel([], [], ["N000"])


@pytest.mark.unit
def test_build_did_panel_length_mismatch_raises(rate_sensitive_panel: SyntheticPanel) -> None:
    """Misaligned ``abnormal_by_event`` / ``surprises`` raise ``ValidationError``."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    with pytest.raises(ValidationError):
        build_did_panel(abn, ["hawkish"], panel.rate_sensitive)


@pytest.mark.unit
def test_build_did_panel_identical_surprises_raise(rate_sensitive_panel: SyntheticPanel) -> None:
    """Coinciding treated/control surprise labels raise ``ValidationError``."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    with pytest.raises(ValidationError):
        build_did_panel(
            abn,
            _surprises_for(panel, windows),
            panel.rate_sensitive,
            treated_surprise="hawkish",
            control_surprise="hawkish",
        )


@pytest.mark.unit
def test_build_did_panel_no_contrast_events_raise(rate_sensitive_panel: SyntheticPanel) -> None:
    """All-neutral surprises leave no observations -> ``InsufficientDataError``."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    neutral = ["neutral"] * len(abn)
    with pytest.raises(InsufficientDataError):
        build_did_panel(abn, neutral, panel.rate_sensitive)


@pytest.mark.unit
def test_build_did_panel_unidentified_cell_raises(rate_sensitive_panel: SyntheticPanel) -> None:
    """A missing treated/post cell (no controls) raises ``InsufficientDataError``."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    # Designate EVERY name rate-sensitive -> no control rows -> (0, *) cells empty.
    all_names = list(panel.returns.columns)
    with pytest.raises(InsufficientDataError):
        build_did_panel(abn, _surprises_for(panel, windows), all_names)


# --------------------------------------------------------------------------- #
# model.py — estimate_did (recovery + inference)                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_estimate_did_recovers_injected_heterogeneity(
    rate_sensitive_panel: SyntheticPanel,
) -> None:
    """The DiD coefficient recovers the injected treated-vs-control heterogeneity.

    On ``rate_sensitive_panel`` the treated names absorb the injected CAR plus a
    hawkish-vs-dovish surprise tilt, so the ``treated x post`` interaction is
    materially positive and significant under clustered SEs.
    """
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    result = estimate_did(dp, cluster="event")
    assert isinstance(result, DiDResult)
    # The injected hawkish-minus-dovish treated tilt is ~ surprise_tilt magnitude;
    # the recovered coefficient is comfortably positive and significant.
    assert result.coef > 0.0
    assert result.t_stat > 2.0
    assert result.p_value < 0.05
    assert result.cluster == "event"
    assert result.n_obs == int(dp.shape[0])
    assert result.n_clusters == int(dp["event"].nunique())


@pytest.mark.unit
def test_estimate_did_coef_equals_cell_means(rate_sensitive_panel: SyntheticPanel) -> None:
    """The DiD coef equals the textbook double-difference of the four cell means."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    result = estimate_did(dp)

    def cell(t: int, p: int) -> float:
        sub = dp[(dp["treated"] == t) & (dp["post"] == p)]
        return float(sub["y"].mean())

    double_diff = (cell(1, 1) - cell(1, 0)) - (cell(0, 1) - cell(0, 0))
    assert result.coef == pytest.approx(double_diff, abs=1e-10)


@pytest.mark.unit
def test_did_drives_heterogeneity_spread_end_to_end(
    rate_sensitive_panel: SyntheticPanel,
) -> None:
    """The DiD result + per-event spreads flow into a positive heterogeneity spread.

    On the (real-effect) ``rate_sensitive_panel`` the treated-minus-control spread
    is positive, so the gross heterogeneity spread is positive — the descriptive
    magnitude the verdict later stresses net of costs.
    """
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    result = estimate_did(dp)
    spreads = _event_spreads(panel, abn)
    spread = heterogeneity_spread(result, spreads)
    assert spread.gross_spread == pytest.approx(float(spreads.mean()))
    assert spread.gross_spread > 0.0
    # The gross spread averages over ALL events (incl. neutral) while the DiD coef
    # contrasts only hawkish-vs-dovish, so they share a sign but need not be equal.
    assert np.sign(spread.gross_spread) == np.sign(result.coef)


@pytest.mark.unit
def test_estimate_did_two_way_cluster(rate_sensitive_panel: SyntheticPanel) -> None:
    """Two-way (event+name) clustering yields the same coef and a valid SE."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    one_way = estimate_did(dp, cluster="event")
    two_way = estimate_did(dp, cluster="event+name")
    # The point estimate is invariant to the clustering scheme.
    assert two_way.coef == pytest.approx(one_way.coef, abs=1e-12)
    assert two_way.cluster == "event+name"
    assert two_way.std_error > 0.0
    # n_clusters is the binding (smaller) cluster dimension.
    assert two_way.n_clusters == min(dp["event"].nunique(), dp["name"].nunique())


@pytest.mark.unit
def test_estimate_did_to_dict_is_json_serializable(rate_sensitive_panel: SyntheticPanel) -> None:
    """``DiDResult.to_dict`` emits only JSON scalars."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    payload = estimate_did(dp).to_dict()
    assert set(payload) == {
        "coef",
        "std_error",
        "t_stat",
        "p_value",
        "n_obs",
        "n_clusters",
        "cluster",
    }
    assert isinstance(payload["n_obs"], int)
    assert isinstance(payload["coef"], float)
    assert payload["cluster"] == "event"


# --------------------------------------------------------------------------- #
# model.py — estimate_did (degenerate-input guards)                           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_estimate_did_rejects_unknown_cluster(rate_sensitive_panel: SyntheticPanel) -> None:
    """An unrecognized clustering scheme raises ``ValidationError``."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    with pytest.raises(ValidationError):
        estimate_did(dp, cluster="firm")


@pytest.mark.unit
def test_estimate_did_rejects_bad_alpha(rate_sensitive_panel: SyntheticPanel) -> None:
    """An out-of-range ``alpha`` raises ``ValidationError``."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    with pytest.raises(ValidationError):
        estimate_did(dp, alpha=1.5)


@pytest.mark.unit
def test_estimate_did_rejects_missing_columns() -> None:
    """A panel missing required columns raises ``ValidationError``."""
    bad = pd.DataFrame({"y": [0.1, 0.2], "treated": [0, 1]})
    with pytest.raises(ValidationError):
        estimate_did(bad)


@pytest.mark.unit
def test_estimate_did_too_few_observations_raises() -> None:
    """Fewer than ``n_params + 1`` rows raises ``InsufficientDataError``."""
    tiny = pd.DataFrame(
        {
            "y": [0.1, 0.2, 0.3, 0.4],
            "treated": [0, 1, 0, 1],
            "post": [0, 0, 1, 1],
            "interaction": [0, 0, 0, 1],
            "event": [0, 0, 1, 1],
            "name": ["A", "B", "A", "B"],
        }
    )
    with pytest.raises(InsufficientDataError):
        estimate_did(tiny)


@pytest.mark.unit
def test_estimate_did_single_cluster_raises() -> None:
    """A panel with one event cluster cannot be clustered -> ``InsufficientDataError``."""
    rng = np.random.default_rng(0)
    n = 40
    treated = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "y": rng.normal(0, 1, n),
            "treated": treated,
            "post": post,
            "interaction": treated * post,
            "event": np.zeros(n, dtype=int),  # ONE cluster
            "name": [f"N{i % 8}" for i in range(n)],
        }
    )
    with pytest.raises(InsufficientDataError):
        estimate_did(df, cluster="event")


@pytest.mark.unit
def test_estimate_did_rank_deficient_design_raises() -> None:
    """A collinear design (no post variation) raises ``InsufficientDataError``."""
    rng = np.random.default_rng(1)
    n = 40
    treated = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "y": rng.normal(0, 1, n),
            "treated": treated,
            "post": np.zeros(n, dtype=int),  # no post variation -> rank deficient
            "interaction": np.zeros(n, dtype=int),
            "event": rng.integers(0, 6, n),
            "name": [f"N{i % 8}" for i in range(n)],
        }
    )
    with pytest.raises(InsufficientDataError):
        estimate_did(df, cluster="event")


@pytest.mark.unit
def test_estimate_did_zero_residual_variance_is_neutral() -> None:
    """A perfectly-fit DiD (zero residuals) returns t=0, p=1 rather than dividing.

    When ``y`` is an EXACT linear function of the regressors the residuals — and
    thus the clustered variance of the interaction — are zero; the degenerate
    guard must return a neutral, finite result.
    """
    rng = np.random.default_rng(2)
    n = 60
    treated = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    interaction = treated * post
    # y is exactly 0.5 + 0.1*treated + 0.2*post + 0.3*interaction (no noise).
    y = 0.5 + 0.1 * treated + 0.2 * post + 0.3 * interaction
    df = pd.DataFrame(
        {
            "y": y,
            "treated": treated,
            "post": post,
            "interaction": interaction,
            "event": rng.integers(0, 8, n),
            "name": [f"N{i % 10}" for i in range(n)],
        }
    )
    result = estimate_did(df, cluster="event")
    assert result.coef == pytest.approx(0.3, abs=1e-9)
    assert result.std_error == pytest.approx(0.0, abs=1e-12)
    assert result.t_stat == 0.0
    assert result.p_value == 1.0


@pytest.mark.unit
def test_build_did_panel_skips_non_finite_cars(rate_sensitive_panel: SyntheticPanel) -> None:
    """A NaN abnormal-return cell is dropped, not propagated into the panel."""
    panel = rate_sensitive_panel
    windows = _windows(panel)
    abn = _abnormal_by_event(panel, windows)
    # Poison one treated name's CAR in the first event with a NaN.
    poisoned = abn[0].copy()
    poisoned.iloc[0, 0] = np.nan
    abn[0] = poisoned
    dp = build_did_panel(abn, _surprises_for(panel, windows), panel.rate_sensitive)
    # No NaN survives into y, and the poisoned (event 0, first name) row is absent.
    assert not dp["y"].isna().any()
    first_name = str(panel.returns.columns[0])
    assert dp[(dp["event"] == 0) & (dp["name"] == first_name)].empty


# --------------------------------------------------------------------------- #
# heterogeneity.py — heterogeneity_spread                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_heterogeneity_spread_charges_round_trip_cost() -> None:
    """The net spread is the gross spread minus the decimal round-trip cost."""
    did = _did_result(coef=0.01)
    samples = np.full(30, 0.01)
    spread = heterogeneity_spread(did, samples, cost_bps=20.0)
    assert isinstance(spread, HeterogeneitySpread)
    assert spread.gross_spread == pytest.approx(0.01)
    # 20 bps == 0.0020 decimal subtracted from each sample.
    assert spread.net_spread == pytest.approx(0.01 - 0.002)
    assert spread.cost_bps == pytest.approx(20.0)


@pytest.mark.unit
def test_heterogeneity_spread_tradable_when_clear() -> None:
    """A large, low-variance positive spread survives costs and reads tradable."""
    did = _did_result(coef=0.02)
    rng = np.random.default_rng(3)
    samples = rng.normal(0.02, 0.001, size=60)  # mean >> cost, tight variance
    spread = heterogeneity_spread(did, samples, cost_bps=20.0)
    assert spread.net_spread > 0.0
    assert spread.is_tradable_spread is True


@pytest.mark.unit
def test_heterogeneity_spread_not_tradable_after_cost() -> None:
    """A small/noisy spread that costs eat into reads NOT tradable (honest null)."""
    did = _did_result(coef=0.001)
    rng = np.random.default_rng(4)
    samples = rng.normal(0.0005, 0.01, size=40)  # mean below the 20bps hurdle
    spread = heterogeneity_spread(did, samples, cost_bps=DEFAULT_COST_BPS)
    assert spread.net_spread < 0.0
    assert spread.is_tradable_spread is False


@pytest.mark.unit
def test_heterogeneity_spread_positive_mean_but_insignificant_not_tradable() -> None:
    """A positive net mean that is statistically indistinct from zero is NOT tradable."""
    did = _did_result(coef=0.005)
    rng = np.random.default_rng(5)
    # Net mean is positive but the noise is huge -> insignificant.
    samples = rng.normal(0.0025, 0.05, size=30)
    spread = heterogeneity_spread(did, samples, cost_bps=DEFAULT_COST_BPS)
    # Even if the net mean happens positive, significance gate must also clear.
    if spread.net_spread > 0.0:
        assert spread.is_tradable_spread is False


@pytest.mark.unit
def test_heterogeneity_spread_constant_samples_not_tradable() -> None:
    """A zero-variance net spread is never spuriously significant (degenerate guard)."""
    did = _did_result(coef=0.01)
    spread = heterogeneity_spread(did, np.full(20, 0.01), cost_bps=20.0)
    # Net spread positive (0.008) but zero variance -> p-value 1.0 -> not tradable.
    assert spread.net_spread > 0.0
    assert spread.is_tradable_spread is False


@pytest.mark.unit
def test_heterogeneity_spread_rejects_negative_cost() -> None:
    """A negative ``cost_bps`` raises ``ValidationError``."""
    did = _did_result()
    with pytest.raises(ValidationError):
        heterogeneity_spread(did, np.full(10, 0.01), cost_bps=-1.0)


@pytest.mark.unit
def test_heterogeneity_spread_rejects_bad_alpha() -> None:
    """An out-of-range ``alpha`` raises ``ValidationError``."""
    did = _did_result()
    with pytest.raises(ValidationError):
        heterogeneity_spread(did, np.full(10, 0.01), alpha=0.0)


@pytest.mark.unit
def test_heterogeneity_spread_empty_samples_raise() -> None:
    """All-NaN / empty ``spread_samples`` raises ``InsufficientDataError``."""
    did = _did_result()
    with pytest.raises(InsufficientDataError):
        heterogeneity_spread(did, np.array([np.nan, np.nan]))


@pytest.mark.unit
def test_heterogeneity_spread_single_sample_not_tradable() -> None:
    """A single net-spread observation cannot be significant -> NOT tradable.

    With one observation there is no variance estimate, so the net-spread test
    returns ``p = 1`` and the spread is flagged untradable even if its mean is
    positive after costs (the degenerate-sample guard).
    """
    did = _did_result(coef=0.05)
    spread = heterogeneity_spread(did, np.array([0.05]), cost_bps=20.0)
    assert spread.net_spread > 0.0  # 0.05 - 0.002 > 0
    assert spread.is_tradable_spread is False


@pytest.mark.unit
def test_heterogeneity_spread_to_dict_is_json_serializable() -> None:
    """``HeterogeneitySpread.to_dict`` emits only JSON scalars."""
    did = _did_result()
    payload = heterogeneity_spread(did, np.full(20, 0.01)).to_dict()
    assert set(payload) == {
        "gross_spread",
        "cost_bps",
        "net_spread",
        "net_pvalue",
        "is_tradable_spread",
    }
    assert isinstance(payload["is_tradable_spread"], bool)
    assert isinstance(payload["net_spread"], float)
    assert isinstance(payload["net_pvalue"], float)


# --------------------------------------------------------------------------- #
# heterogeneity.py — describe_heterogeneity                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_describe_heterogeneity_is_honest_non_promotional() -> None:
    """The interpretation frames the result as heterogeneity, not a tradable alpha."""
    did = _did_result(coef=0.012, t_stat=3.5)
    spread = heterogeneity_spread(did, np.full(20, 0.01))  # not tradable (constant)
    text = describe_heterogeneity(did, spread)
    lowered = text.lower()
    assert "heterogeneity" in lowered
    assert "not a tradable causal alpha" in lowered
    assert "does not survive transaction costs" in lowered


@pytest.mark.unit
def test_describe_heterogeneity_reports_direction() -> None:
    """A negative coefficient is described as 'less', a positive one as 'more'."""
    did_neg = _did_result(coef=-0.01, t_stat=-2.5)
    spread = heterogeneity_spread(did_neg, np.full(20, -0.01))
    assert "less than controls" in describe_heterogeneity(did_neg, spread)
