"""Parity tests for the DiD clustered standard errors.

The one-way ``"event"`` clustered standard error from
:func:`fedcausal.did.model.estimate_did` is validated against an independent
``statsmodels`` OLS fit with ``cov_type="cluster"`` (the same Liang-Zeger sandwich
with the same finite-sample correction), to 1e-10. The DiD point estimate is also
checked against ``statsmodels`` OLS to machine precision.

``statsmodels`` is used ONLY as the parity oracle here; the production code is
pure numpy and never imports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcausal.did.model import build_did_panel, estimate_did

sm = pytest.importorskip("statsmodels.api")


def _synthetic_did_panel(seed: int = 11, n_events: int = 12, n_names: int = 24) -> pd.DataFrame:
    """A small, well-identified DiD panel (per-(event, name) CAR observations)."""
    rng = np.random.default_rng(seed)
    rate_sensitive = [f"N{i:03d}" for i in range(n_names // 2)]
    abnormal_by_event: list[pd.DataFrame] = []
    surprises: list[str] = []
    tickers = [f"N{i:03d}" for i in range(n_names)]
    for ev in range(n_events):
        surprise = "hawkish" if ev % 2 == 0 else "dovish"
        surprises.append(surprise)
        # Three event-relative days; treated names get a sign-dependent tilt.
        base = rng.normal(0.0, 0.01, size=(3, n_names))
        ar = pd.DataFrame(base, index=[-1, 0, 1], columns=tickers)
        tilt = 0.004 if surprise == "hawkish" else -0.001
        ar[rate_sensitive] += tilt / 3.0
        abnormal_by_event.append(ar)
    return build_did_panel(abnormal_by_event, surprises, rate_sensitive)  # type: ignore[arg-type]


@pytest.mark.parity
def test_did_point_estimate_matches_statsmodels_ols() -> None:
    """The DiD interaction coef equals a statsmodels OLS coefficient (1e-12)."""
    panel = _synthetic_did_panel()
    result = estimate_did(panel, cluster="event")

    y = panel["y"].to_numpy(dtype=float)
    x = sm.add_constant(
        np.column_stack(
            [
                panel["treated"].to_numpy(dtype=float),
                panel["post"].to_numpy(dtype=float),
                panel["interaction"].to_numpy(dtype=float),
            ]
        )
    )
    ref = sm.OLS(y, x).fit()
    # The interaction is the last regressor.
    assert result.coef == pytest.approx(float(ref.params[-1]), abs=1e-12)


@pytest.mark.parity
def test_did_clustered_se_matches_statsmodels_cluster() -> None:
    """The one-way clustered SE matches statsmodels ``cov_type='cluster'`` (1e-10)."""
    panel = _synthetic_did_panel()
    result = estimate_did(panel, cluster="event")

    y = panel["y"].to_numpy(dtype=float)
    x = sm.add_constant(
        np.column_stack(
            [
                panel["treated"].to_numpy(dtype=float),
                panel["post"].to_numpy(dtype=float),
                panel["interaction"].to_numpy(dtype=float),
            ]
        )
    )
    groups = panel["event"].to_numpy()
    ref = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    # SE of the interaction term (last regressor) must agree to 1e-10.
    assert result.std_error == pytest.approx(float(ref.bse[-1]), abs=1e-10)
    # And the number of clusters matches the number of distinct events.
    assert result.n_clusters == int(np.unique(groups).size)


@pytest.mark.parity
def test_did_clustered_se_matches_statsmodels_across_seeds() -> None:
    """Clustered-SE parity holds across several independent synthetic panels."""
    for seed in (1, 7, 21, 99):
        panel = _synthetic_did_panel(seed=seed, n_events=14, n_names=30)
        result = estimate_did(panel, cluster="event")

        y = panel["y"].to_numpy(dtype=float)
        x = sm.add_constant(
            np.column_stack(
                [
                    panel["treated"].to_numpy(dtype=float),
                    panel["post"].to_numpy(dtype=float),
                    panel["interaction"].to_numpy(dtype=float),
                ]
            )
        )
        groups = panel["event"].to_numpy()
        ref = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
        assert result.std_error == pytest.approx(float(ref.bse[-1]), abs=1e-10), f"seed={seed}"
        assert result.coef == pytest.approx(float(ref.params[-1]), abs=1e-12), f"seed={seed}"
