"""Unit tests for the synthetic event panel + real-data loaders (group ``data``).

Coverage:

- determinism: the seeded synthetic generators are byte-identical across calls;
  distinct seeds diverge.
- ground truth: a panel injected with a KNOWN CAR carries that effect in the
  rate-sensitive names' event windows, the estimation window is effect-free
  (so an estimation-window-only market model is unbiased), and the pure-noise
  panel injects nothing.
- FRED: the keyless ``fredgraph`` CSV parser drops ``.`` missing markers and
  parses both the modern (``observation_date``) and legacy (``DATE``) headers;
  ``load_surprise_labels`` is a pass-through unless ``use_fred`` and degrades on
  failure; the release-date lag reads the last value on/before a meeting.
- PIT / returns: ``fetch_polygon_returns`` validates inputs and converts adjusted
  closes to simple returns with ``pct_change(fill_method=None)`` (no ffill) via a
  monkeypatched provider; the loader defaults to synthetic and never hard-fails.
- import purity: importing the data modules pulls in no network/heavy module.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fedcausal._exceptions import ValidationError
from fedcausal.data import loaders
from fedcausal.data.loaders import (
    fetch_polygon_returns,
    load_event_panel,
    load_surprise_labels,
    parse_fredgraph_csv,
)
from fedcausal.data.synthetic import (
    SyntheticPanel,
    pure_noise_panel,
    rate_sensitive_panel,
    synthetic_event_panel,
)
from fedcausal.events.calendar import FOMCEvent

# --------------------------------------------------------------------------- #
# Synthetic panel: determinism                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_synthetic_panel_is_deterministic() -> None:
    """The same seed yields byte-identical returns, market, and metadata."""
    a = synthetic_event_panel(seed=11)
    b = synthetic_event_panel(seed=11)
    pd.testing.assert_frame_equal(a.returns, b.returns)
    pd.testing.assert_series_equal(a.market, b.market)
    assert a.announcement_dates == b.announcement_dates
    assert a.surprises == b.surprises
    assert a.rate_sensitive == b.rate_sensitive
    assert a.injected_car == b.injected_car


@pytest.mark.unit
def test_distinct_seeds_diverge() -> None:
    """Different seeds produce genuinely different return panels."""
    a = synthetic_event_panel(seed=1)
    b = synthetic_event_panel(seed=2)
    assert not np.allclose(a.returns.to_numpy(), b.returns.to_numpy())


@pytest.mark.unit
def test_panel_shape_and_grid_are_consistent() -> None:
    """The panel is a clean wide float64 frame on a business-day grid."""
    panel = synthetic_event_panel(n_names=12, n_events=5, seed=3)
    assert isinstance(panel, SyntheticPanel)
    assert panel.returns.shape[1] == 12
    assert len(panel.announcement_dates) == 5
    assert len(panel.surprises) == 5
    assert str(panel.returns.to_numpy().dtype) == "float64"
    assert isinstance(panel.returns.index, pd.DatetimeIndex)
    assert panel.returns.index.is_monotonic_increasing
    # market aligns with returns row index
    assert panel.market.index.equals(panel.returns.index)
    # every announcement date is present on the trading grid
    grid = {ts.date() for ts in panel.returns.index}
    for d in panel.announcement_dates:
        assert d in grid
    # rate-sensitive names are a strict, non-empty subset of the columns
    cols = set(panel.returns.columns)
    assert set(panel.rate_sensitive) <= cols
    assert 0 < len(panel.rate_sensitive) < len(cols)


@pytest.mark.unit
def test_to_dict_is_json_summary() -> None:
    """``to_dict`` returns scalar metadata without the bulk panels."""
    panel = synthetic_event_panel(n_names=8, n_events=4, seed=5)
    summary = panel.to_dict()
    assert summary["n_names"] == 8
    assert summary["n_events"] == 4
    assert summary["seed"] == 5
    assert "returns" not in summary and "market" not in summary


# --------------------------------------------------------------------------- #
# Synthetic panel: ground truth (known-CAR injection recoverable)             #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_known_car_is_injected_into_rate_sensitive_names() -> None:
    """The injected CAR appears in rate-sensitive event windows, not in controls.

    We recover the abnormal return as (actual - one-factor expectation) using the
    panel's own market series and an OLS market-model fit on a clean pre-event
    estimation window, then sum over each event window. Averaged across events,
    the rate-sensitive CAR concentrates on the injected value while the control
    CAR concentrates on zero.
    """
    injected = 0.02
    k = 1
    panel = synthetic_event_panel(
        n_names=20,
        n_events=20,
        injected_car=injected,
        event_half_width=k,
        noise_scale=0.004,
        seed=7,
    )
    grid = panel.returns.index
    market = panel.market.to_numpy()

    rs_cars: list[float] = []
    ctrl_cars: list[float] = []
    for ann in panel.announcement_dates:
        pos = int(grid.get_loc(pd.Timestamp(ann)))
        est_lo, est_hi = pos - 10 - 60, pos - 10 - 1  # clean pre-event window
        ev_lo, ev_hi = pos - k, pos + k
        x = market[est_lo : est_hi + 1]
        # design = [market, 1] so the lstsq solution is [beta, alpha]
        x_design = np.column_stack([x, np.ones_like(x)])
        market_ev = market[ev_lo : ev_hi + 1]
        for name in panel.returns.columns:
            col = panel.returns[name].to_numpy()
            beta, alpha = np.linalg.lstsq(x_design, col[est_lo : est_hi + 1], rcond=None)[0]
            expected = alpha + beta * market_ev
            car = float(np.sum(col[ev_lo : ev_hi + 1] - expected))
            if name in set(panel.rate_sensitive):
                rs_cars.append(car)
            else:
                ctrl_cars.append(car)

    rs_mean = float(np.mean(rs_cars))
    ctrl_mean = float(np.mean(ctrl_cars))
    # Rate-sensitive names recover the injected CAR within tolerance; the
    # estimation-window-only model leaves controls near zero.
    assert rs_mean == pytest.approx(injected, abs=3e-3)
    assert abs(ctrl_mean) < 3e-3
    assert rs_mean - ctrl_mean > injected / 2


@pytest.mark.unit
def test_estimation_window_is_effect_free() -> None:
    """No injected effect contaminates the pre-event estimation window.

    For each event, the rate-sensitive names' mean return over a clean pre-event
    slice is statistically indistinguishable from the control names' — the
    injected effect lives ONLY in the [-k, +k] event window, so an
    estimation-window-only market model is unbiased.
    """
    panel = synthetic_event_panel(n_names=20, n_events=16, injected_car=0.03, seed=9)
    grid = panel.returns.index
    rs = set(panel.rate_sensitive)
    rs_means: list[float] = []
    ctrl_means: list[float] = []
    for ann in panel.announcement_dates:
        pos = int(grid.get_loc(pd.Timestamp(ann)))
        est = slice(pos - 10 - 60, pos - 10)  # strictly before the event window
        block = panel.returns.iloc[est]
        rs_means.append(float(block[[c for c in block.columns if c in rs]].to_numpy().mean()))
        ctrl_means.append(float(block[[c for c in block.columns if c not in rs]].to_numpy().mean()))
    # The treated/control gap in the estimation window is ~0 (no leakage).
    assert abs(np.mean(rs_means) - np.mean(ctrl_means)) < 1e-3


@pytest.mark.unit
def test_pure_noise_panel_injects_nothing() -> None:
    """The honest-null control injects a zero CAR and is genuinely effect-free."""
    panel = pure_noise_panel(seed=7)
    assert panel.injected_car == 0.0
    # Cross-event mean return on announcement days is tiny (pure idiosyncratic).
    grid = panel.returns.index
    day_means = [
        float(panel.returns.iloc[int(grid.get_loc(pd.Timestamp(ann)))].mean())
        for ann in panel.announcement_dates
    ]
    assert abs(np.mean(day_means)) < 5e-3


@pytest.mark.unit
def test_pure_noise_ignores_injected_car_kwarg() -> None:
    """``pure_noise_panel`` forces ``injected_car=0`` even if one is passed."""
    panel = pure_noise_panel(seed=7, injected_car=0.5)
    assert panel.injected_car == 0.0


@pytest.mark.unit
def test_rate_sensitive_panel_amplifies_heterogeneity() -> None:
    """The rate-sensitive preset injects a (larger) default CAR for the DiD."""
    panel = rate_sensitive_panel(seed=7)
    assert panel.injected_car > 0.0
    assert len(panel.rate_sensitive) >= 1
    # all three surprise signs are present across the default 16-event calendar
    assert {"hawkish", "dovish", "neutral"} <= set(panel.surprises)


# --------------------------------------------------------------------------- #
# Synthetic panel: validation                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_names": 1},
        {"n_events": 0},
        {"event_half_width": 0},
        {"event_half_width": 999},
        {"estimation_window": 1},
        {"spacing": 2, "event_half_width": 2},
        {"rate_sensitive_frac": 0.0},
        {"rate_sensitive_frac": 1.0},
        {"noise_scale": 0.0},
        {"noise_scale": -1.0},
    ],
)
def test_synthetic_panel_rejects_bad_params(kwargs: dict[str, object]) -> None:
    """Out-of-range sizing parameters raise ``ValidationError``."""
    with pytest.raises(ValidationError):
        synthetic_event_panel(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# FRED keyless CSV parsing + release-date lag                                 #
# --------------------------------------------------------------------------- #

_MODERN_CSV = "observation_date,DFEDTARU\n2022-03-16,0.50\n2022-05-04,1.00\n2022-06-15,1.75\n"
_LEGACY_CSV = "DATE,DFEDTARU\n2022-03-16,0.50\n2022-05-04,1.00\n"
_CSV_WITH_MISSING = "observation_date,DFF\n2022-03-14,0.08\n2022-03-15,.\n2022-03-16,0.33\n"


@pytest.mark.unit
def test_parse_fredgraph_csv_modern_header() -> None:
    """The modern ``observation_date`` header parses to a date-indexed float series."""
    series = parse_fredgraph_csv(_MODERN_CSV)
    assert list(series.to_numpy()) == [0.50, 1.00, 1.75]
    assert series.index[0] == pd.Timestamp("2022-03-16")
    assert series.name == "DFEDTARU"
    assert str(series.to_numpy().dtype) == "float64"


@pytest.mark.unit
def test_parse_fredgraph_csv_legacy_header() -> None:
    """The legacy ``DATE`` header is parsed identically (column 0 = date)."""
    series = parse_fredgraph_csv(_LEGACY_CSV)
    assert len(series) == 2
    assert series.index.is_monotonic_increasing


@pytest.mark.unit
def test_parse_fredgraph_csv_drops_missing_markers() -> None:
    """FRED's ``.`` missing markers are dropped, not parsed as values."""
    series = parse_fredgraph_csv(_CSV_WITH_MISSING)
    assert len(series) == 2  # the "." row is removed
    assert pd.Timestamp("2022-03-15") not in series.index
    assert list(series.to_numpy()) == [0.08, 0.33]


@pytest.mark.unit
def test_release_date_lag_uses_last_value_on_or_before() -> None:
    """The surprise change reads the last published value on/before each meeting.

    The DFEDTARU step from 0.50 (2022-03-16) to 1.00 (2022-05-04) is +50 bps; a
    meeting dated a few days AFTER the step still reads the stepped value (no
    future revision, point-in-time release awareness).
    """
    series = parse_fredgraph_csv(_MODERN_CSV)
    change = loaders._surprise_change_at(
        series, current=date(2022, 5, 6), previous=date(2022, 3, 18)
    )
    assert change == pytest.approx(50.0)  # (1.00 - 0.50) * 100 bps
    # With no predecessor the change is defined as 0.
    assert loaders._surprise_change_at(series, current=date(2022, 5, 6), previous=None) == 0.0


@pytest.mark.unit
def test_fetch_fred_series_rejects_empty_id() -> None:
    """An empty/whitespace series id raises ``ValidationError`` before any I/O."""
    with pytest.raises(ValidationError):
        loaders.fetch_fred_series("   ")


@pytest.mark.unit
def test_parse_fredgraph_csv_rejects_single_column() -> None:
    """A one-column CSV (no value column) raises ``RuntimeError``."""
    with pytest.raises(RuntimeError):
        parse_fredgraph_csv("observation_date\n2022-03-16\n")


@pytest.mark.unit
def test_surprise_change_requires_observation_before_each_meeting() -> None:
    """The release-date lookup raises if no observation precedes a meeting date."""
    series = parse_fredgraph_csv(_MODERN_CSV)  # first obs 2022-03-16
    # current before any observation -> RuntimeError
    with pytest.raises(RuntimeError):
        loaders._surprise_change_at(series, current=date(2000, 1, 1), previous=None)
    # current ok but previous before any observation -> RuntimeError
    with pytest.raises(RuntimeError):
        loaders._surprise_change_at(series, current=date(2022, 6, 15), previous=date(2000, 1, 1))


class _FakeResponse:
    """Minimal stand-in for an ``httpx.Response`` (text + raise_for_status)."""

    def __init__(self, text: str, *, ok: bool = True) -> None:
        self.text = text
        self._ok = ok

    def raise_for_status(self) -> None:
        if not self._ok:
            import httpx

            raise httpx.HTTPError("boom")


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    """Patch ``httpx.get`` (the lazily-imported client) to return ``response``."""
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: response)


@pytest.mark.unit
def test_fetch_fred_series_parses_mocked_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mocked keyless CSV download parses to a date-indexed series with bounds."""
    _install_fake_httpx(monkeypatch, _FakeResponse(_MODERN_CSV))
    series = loaders.fetch_fred_series(
        loaders.FRED_TARGET_UPPER, start=date(2022, 4, 1), end=date(2022, 12, 31)
    )
    # start bound drops the 2022-03-16 row; two observations remain.
    assert list(series.to_numpy()) == [1.00, 1.75]
    assert series.index.min() >= pd.Timestamp("2022-04-01")


@pytest.mark.unit
def test_fetch_fred_series_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HTTP error is wrapped in ``RuntimeError`` so callers can fall back."""
    _install_fake_httpx(monkeypatch, _FakeResponse("", ok=False))
    with pytest.raises(RuntimeError):
        loaders.fetch_fred_series("DFF")


@pytest.mark.unit
def test_fetch_fred_series_raises_when_empty_after_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload emptied by the date bounds raises ``RuntimeError`` (no fall-through)."""
    _install_fake_httpx(monkeypatch, _FakeResponse(_MODERN_CSV))
    with pytest.raises(RuntimeError):
        loaders.fetch_fred_series(loaders.FRED_TARGET_UPPER, start=date(2030, 1, 1))


@pytest.mark.unit
def test_load_surprise_labels_refreshes_when_classifier_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When FRED and the classifier both work, labels are re-derived PIT.

    Stubs both the FRED series and ``classify_surprise`` (still a kernel stub in
    this group's slice) so the refresh branch runs end-to-end and re-derives the
    signed rate change at each meeting's release date.
    """
    series = pd.Series(
        [0.25, 0.50, 1.00, 5.00],
        index=pd.to_datetime(["2022-01-26", "2022-03-16", "2022-05-04", "2024-09-18"]),
        name="DFEDTARU",
    )
    import fedcausal.events.calendar as calendar_mod

    monkeypatch.setattr(loaders, "fetch_fred_series", lambda *a, **k: series)
    monkeypatch.setattr(
        calendar_mod,
        "classify_surprise",
        lambda bps: "hawkish" if bps > 0 else "dovish" if bps < 0 else "neutral",
    )
    out = load_surprise_labels(_committed_events(), use_fred=True)
    # First meeting has no predecessor (0 bps -> neutral); +50 then +400 bps.
    assert [ev.surprise for ev in out] == ["neutral", "hawkish", "hawkish"]
    # PIT: 2024-09-18 reads 5.00 vs the prior meeting's 1.00 -> +400 bps.
    assert out[-1].rate_change_bps == pytest.approx(400.0)


# --------------------------------------------------------------------------- #
# load_surprise_labels                                                         #
# --------------------------------------------------------------------------- #


def _committed_events() -> list[FOMCEvent]:
    return [
        FOMCEvent(date(2022, 3, 16), 0.50, 25.0, "hawkish"),
        FOMCEvent(date(2022, 5, 4), 1.00, 50.0, "hawkish"),
        FOMCEvent(date(2024, 9, 18), 5.00, -50.0, "dovish"),
    ]


@pytest.mark.unit
def test_load_surprise_labels_passthrough_when_not_use_fred() -> None:
    """Without ``use_fred`` the events are returned unchanged (no network)."""
    events = _committed_events()
    out = load_surprise_labels(events, use_fred=False)
    assert out == events
    assert out is not events  # a fresh list, not the same object


@pytest.mark.unit
def test_load_surprise_labels_degrades_on_fred_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FRED failure degrades to the committed labels rather than raising."""

    def _boom(*_args: object, **_kwargs: object) -> pd.Series:
        raise RuntimeError("network down")

    monkeypatch.setattr(loaders, "fetch_fred_series", _boom)
    events = _committed_events()
    out = load_surprise_labels(events, use_fred=True)
    assert out == events


@pytest.mark.unit
def test_load_surprise_labels_refreshes_from_fred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``use_fred`` and a (stubbed) series, labels are re-derived PIT."""
    series = pd.Series(
        [0.25, 0.50, 1.00, 5.00],
        index=pd.to_datetime(["2022-01-26", "2022-03-16", "2022-05-04", "2024-09-18"]),
        name="DFEDTARU",
    )
    monkeypatch.setattr(loaders, "fetch_fred_series", lambda *a, **k: series)
    # classify_surprise is still a stub (NotImplementedError); the function must
    # degrade gracefully to the committed labels on that failure.
    out = load_surprise_labels(_committed_events(), use_fred=True)
    assert len(out) == 3


# --------------------------------------------------------------------------- #
# Polygon PIT returns                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_fetch_polygon_returns_validates_inputs() -> None:
    """Empty tickers / non-increasing dates raise ``ValidationError`` before I/O."""
    with pytest.raises(ValidationError):
        fetch_polygon_returns([], date(2022, 1, 1), date(2022, 2, 1))
    with pytest.raises(ValidationError):
        fetch_polygon_returns(["AAPL"], date(2022, 2, 1), date(2022, 1, 1))


@pytest.mark.unit
def test_fetch_polygon_returns_uses_pct_change_no_ffill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjusted closes become simple returns with no forward-fill before diffing.

    A monkeypatched provider returns a price panel with a gap (NaN) for one
    ticker; ``pct_change(fill_method=None)`` must NOT manufacture a spurious zero
    return across the gap (the post-gap pct_change is NaN, not 0).
    """
    idx = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"])
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0, 103.0],
            "MSFT": [200.0, np.nan, 202.0, 204.0],
        },
        index=idx,
    )

    class _FakeProvider:
        def __init__(self, *_a: object, **_k: object) -> None: ...

        def fetch(self, _tickers: list[str], _start: date, _end: date) -> pd.DataFrame:
            return prices

    import fedcausal.data_providers.polygon as polygon_mod

    monkeypatch.setattr(polygon_mod, "PolygonProvider", _FakeProvider)
    out = fetch_polygon_returns(["AAPL", "MSFT"], date(2022, 1, 1), date(2022, 2, 1))

    # Leading row dropped; AAPL is a clean 1% step.
    assert out.shape[0] == 3
    assert out["AAPL"].iloc[0] == pytest.approx(0.01)
    # The return across the MSFT NaN gap is NaN (no ffill-manufactured zero).
    assert bool(out["MSFT"].isna().any())
    assert not (out["MSFT"].fillna(999.0) == 0.0).any()


# --------------------------------------------------------------------------- #
# load_event_panel                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_event_panel_synthetic_default() -> None:
    """The default returns the seeded synthetic panel and reports its source."""
    panel, source = load_event_panel(seed=7)
    assert source == "synthetic"
    assert isinstance(panel, SyntheticPanel)
    # deterministic vs the direct generator
    ref = synthetic_event_panel(seed=7)
    pd.testing.assert_frame_equal(panel.returns, ref.returns)


@pytest.mark.unit
def test_load_event_panel_degrades_to_synthetic_on_real_data_failure() -> None:
    """``fred+polygon`` degrades to synthetic when the real path fails (no key/net).

    The calendar/classifier are still stubs and there is no Polygon key in CI, so
    the real-data path must NEVER hard-fail; it returns the synthetic panel.
    """
    panel, source = load_event_panel(data_source_pref="fred+polygon", seed=7)
    assert source == "synthetic"
    assert isinstance(panel, SyntheticPanel)


@pytest.mark.unit
def test_load_event_panel_real_data_success_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mocked calendar + Polygon path assembles a ``fred+polygon`` panel.

    Stubs ``load_fomc_calendar`` and ``fetch_polygon_returns`` so the real-data
    assembly branch runs offline: the panel carries the FOMC dates/surprises, an
    equal-weight market proxy, and a NaN ``injected_car`` (no ground truth on
    real data).
    """
    import fedcausal.events.calendar as calendar_mod

    events = _committed_events()
    monkeypatch.setattr(calendar_mod, "load_fomc_calendar", lambda **k: events)

    idx = pd.date_range("2022-03-01", periods=10, freq="B")
    returns = pd.DataFrame(
        {"AAPL": np.linspace(0.0, 0.01, 10), "MSFT": np.linspace(0.0, -0.01, 10)},
        index=idx,
    )
    monkeypatch.setattr(loaders, "fetch_polygon_returns", lambda *a, **k: returns)

    panel, source = load_event_panel(
        data_source_pref="fred+polygon", tickers=["AAPL", "MSFT"], seed=7
    )
    assert source == "fred+polygon"
    assert panel.announcement_dates == [ev.announcement_date for ev in events]
    assert panel.surprises == [ev.surprise for ev in events]
    assert list(panel.market) == pytest.approx(list(returns.mean(axis=1)))
    assert np.isnan(panel.injected_car)
    assert panel.meta["source"] == "fred+polygon"


@pytest.mark.unit
def test_load_event_panel_real_data_empty_returns_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty Polygon panel degrades to synthetic (never hard-fail)."""
    import fedcausal.events.calendar as calendar_mod

    monkeypatch.setattr(calendar_mod, "load_fomc_calendar", lambda **k: _committed_events())
    monkeypatch.setattr(loaders, "fetch_polygon_returns", lambda *a, **k: pd.DataFrame())
    panel, source = load_event_panel(data_source_pref="fred+polygon", tickers=["AAPL"], seed=7)
    assert source == "synthetic"
    assert isinstance(panel, SyntheticPanel)


# --------------------------------------------------------------------------- #
# import purity                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_data_modules_import_is_side_effect_free() -> None:
    """Importing the data modules pulls in no network/heavy module and no network."""
    code = (
        "import sys\n"
        "import fedcausal.data.synthetic\n"
        "import fedcausal.data.loaders\n"
        "forbidden = ['httpx', 'statsmodels', 'plotly', 'typer', 'torch', 'onnxruntime']\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, 'data import leaked: ' + ', '.join(leaked)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout
