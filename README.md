# fed-causal

> Leakage-free causal inference on FOMC announcements — an event study of
> cumulative abnormal returns, stress-tested with placebo-date randomization,
> HAC/clustered standard errors, a multiple-testing correction, and a
> rate-sensitivity difference-in-differences. **Torch-free** (numpy/scipy/statsmodels).

[![CI](https://github.com/FatihHekim0glu/fed-causal/actions/workflows/ci.yml/badge.svg)](https://github.com/FatihHekim0glu/fed-causal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Honest headline

Around FOMC announcements, cumulative abnormal returns (CAR) show a **transient
average move**. But after **placebo-date randomization** + **HAC standard errors**
+ a **multiple-testing correction**, the effect is **NOT** a robust, exploitable
cross-sectional alpha — it is honest cross-sectional **heterogeneity**
(rate-sensitive names move more, mechanically), statistically fragile and **not
tradable net of costs**.

The verdict the repository emits — `fed_effect_is_tradable` — is a **pure
function** of four independent lines of evidence and reads **`False`** unless
**all** clear:

1. the observed CAR is significant against the **placebo-date null** (the primary
   significance source — never a raw t-stat alone), **and**
2. the mean CAR is significant under a **HAC / Newey-West** standard error, **and**
3. at least one specification **survives the multiple-testing correction** across
   the full grid (windows × models × surprise subsets), **and**
4. the rate-sensitivity DiD implies a long/short spread that is still positive and
   significant **net of transaction costs**.

> The deliverable is the rigorous, leakage-free causal-inference stack, **not a
> profit claim.** SCM/DiD are framed as *descriptive heterogeneity*, never alpha.

## What it does

- **Event study** (Brown-Warner 1985, MacKinlay 1997): expected returns from a
  market model (or mean-adjusted model) fit on the **estimation window only**;
  abnormal returns = actual − expected; CAR over the event window `[-k, +k]`.
- **Tests:** cross-sectional t, the **Boehmer-Musumeci-Poulsen (1991)**
  standardized-residual statistic, and **HAC / clustered** standard errors.
- **Placebo-date randomization:** re-run the CAR on random non-event dates
  (excluding every real event window) to build the honest null distribution; the
  observed-vs-placebo percentile is the headline significance.
- **Difference-in-differences:** rate-sensitive ("treated") vs. control names
  around hawkish vs. dovish surprises, with **clustered** standard errors.
- **Multiple testing:** Benjamini-Hochberg / Romano-Wolf across the full
  specification grid, with an **honest `n_tests`** count.
- **Pure verdict:** `fed_effect_is_tradable`, derived (not narrated) from the four
  conditions above.

## Data

- **FOMC announcement dates** are public and ship committed
  (`src/fedcausal/events/calendar_data.py`) with a target-rate snapshot used to
  classify each meeting's surprise (hawkish/dovish/neutral) from the **rate-change
  sign available at the announcement** — no future revisions, no pre-announcement
  signal.
- The deployed default runs on a **seeded synthetic event panel** with a KNOWN
  injected CAR + rate-sensitivity heterogeneity, so the placebo/HAC/DiD machinery
  is validated against ground truth (a known effect is recovered; a pure-noise
  panel yields `fed_effect_is_tradable = False`).
- The optional real-data path uses **keyless FRED** (the `fredgraph` CSV endpoint,
  no API key) for the rate series and **Polygon** (point-in-time universe) for the
  single-name cross-section, degrading to the committed/synthetic panel on any
  failure.

## Install

```bash
uv venv
uv pip install -e ".[data,viz,cli,dev]"
```

Torch-free and lean: there is no `[all]` extra and no `torch`/`onnx`/`sklearn`.

## Validation

### Deployed-default result (the honest null)

These are the metrics the hosted tool serves by default — the seeded synthetic
event panel scored with `event_window=1` (`[-1, +1]`), `estimation_window=120`,
`model="market"`, `surprise="all"`, `n_placebo=500`, `seed=7`. They are committed
in [`src/fedcausal/artifacts/reference.json`](src/fedcausal/artifacts/reference.json)
and held in lock-step with the live `run_analysis` output by a regression test.

| Metric | Value | Reads as |
| --- | ---: | --- |
| Mean CAR | **+1.028%** (`0.010282`) | A real, market-wide transient move around the announcement. |
| CAR _t_-stat | **13.98** | Large in isolation — but a raw _t_-stat is **not** the verdict. |
| HAC _p_-value (Newey-West) | **9.6e-70** | The mean CAR survives a HAC-robust standard error. |
| Placebo percentile | **100.0** (placebo _p_ = 0.00) | The observed CAR sits at the top of the placebo-date null. |
| DiD coefficient | **+0.225%** (`0.002251`, _p_ = 0.66) | The rate-sensitivity (hawkish-vs-dovish) heterogeneity — small and insignificant. |
| Net-of-cost DiD spread | **−0.091%** (`−0.000911`) | The oracle long/short is **negative** after costs — not tradable. |
| `n_tests` (spec grid) | **18** | Honest multiple-testing denominator (3 widths × 2 models × 3 surprise subsets). |
| **`fed_effect_is_tradable`** | **`false`** | The CAR is a market-wide move, **not** a net-of-cost tradable cross-sectional spread, so the pure verdict is `false`. |

The takeaway is the deliverable: even when the CAR is highly significant against
the placebo null **and** HAC-robust, the move is a **market-wide event-window
drift** plus small **cross-sectional heterogeneity** (rate-sensitive names move
with the surprise sign) — **not** a long/short alpha that survives transaction
costs. The net-of-cost spread is **negative**, so the verdict fails on the
tradability leg. This holds **structurally** across seeds (the injected
rate-sensitivity edge is calibrated below the round-trip cost), not by a knife-edge
_p_-value. The verdict is a pure function — it is **not narrated down to `false`**,
it is *derived* `false`.

The pure-noise control panel fails the placebo + tradability legs (placebo _p_ =
0.28, HAC _p_ = 0.16, net spread negative) and is likewise
`fed_effect_is_tradable = false`, confirming the machinery does not manufacture an
effect where none was injected.

### Correctness gates

Every gate below is an executed test (`tests/{parity,property,regression,integration}`),
not a claim. Run them with `uv run pytest`.

| Gate | What it asserts | Test |
| --- | --- | --- |
| Market-model vs statsmodels | Estimation-window-only market-model abnormal returns match a statsmodels OLS reference | `parity/test_parity_eventstudy.py::test_market_model_abnormal_returns_vs_statsmodels_ols` |
| HAC parity (1e-10) | Newey-West HAC standard error matches the reused `evaluation/hac.py` reference to `1e-10` | `parity/test_parity_eventstudy.py::test_hac_se_matches_reference_to_1e_10` |
| BMP statistic | Boehmer-Musumeci-Poulsen standardized-residual statistic matches a hand reference | `parity/test_parity_eventstudy.py::test_bmp_statistic_vs_hand_reference` |
| BH FWER | Benjamini-Hochberg adjusted _p_-values match statsmodels and are monotone | `parity/test_parity_multiple_testing.py::test_benjamini_hochberg_matches_statsmodels` |
| Romano-Wolf FWER | Romano-Wolf stepdown matches a hand reference and controls FWER on the global null | `parity/test_parity_multiple_testing.py::test_romano_wolf_controls_fwer_on_global_null` |
| DiD vs statsmodels | DiD point estimate and clustered SE match statsmodels OLS / cluster-robust covariance | `parity/test_parity_did.py::test_did_clustered_se_matches_statsmodels_cluster` |
| Estimation/event no-overlap | The estimation and event windows never overlap or straddle (property) | `property/test_events_windows.py::test_estimation_and_event_windows_never_overlap` |
| Placebo excludes events | Placebo dates exclude every real event window — no contamination (property) | `property/test_eventstudy_invariants.py::test_placebo_dates_exclude_every_real_event_window` |
| Beta invariance | Perturbing event-window returns leaves the fitted market-model betas byte-identical (property) | `property/test_eventstudy_invariants.py::test_market_model_beta_invariant_to_event_window_perturbation` |
| Known-CAR recovery | The injected CAR is market-wide and recovered within tolerance on BOTH groups (treated `0.0096`, control `0.0107` vs injected `0.01`); the treated-minus-control gap is only the small tilt | `regression/test_regression_honest_null.py::test_known_car_recovered_within_tolerance` |
| Pure-noise honest-null | A pure-noise panel yields `fed_effect_is_tradable = False` after placebo + multiple testing, deterministically across `PYTHONHASHSEED` | `regression/test_regression_honest_null.py::test_pure_noise_panel_is_not_tradable` |

## Reproduce

Lean install (no `torch`/`onnx`/`sklearn`), then run the CLI and the gates:

```bash
# 1. Lean environment
uv venv
uv pip install -e ".[data,viz,cli,dev]"

# 2. CLI (all default to the seeded synthetic panel — no network)
uv run fedcausal eventstudy --event-window 1 --estimation-window 120 --model market
uv run fedcausal placebo    --n-placebo 500 --seed 7
uv run fedcausal did        --event-window 1 --estimation-window 120

# 3. Regenerate the committed deployed-default reference (drift-guarded in CI)
uv run python scripts/build_reference.py        # writes src/fedcausal/artifacts/reference.json
uv run python scripts/build_reference.py --check # CI drift guard (no write)

# 4. Quality gates (all green: ruff, strict mypy, pytest-cov >= 85%)
uv run ruff check .
uv run mypy src/fedcausal
uv run pytest --cov=fedcausal --cov-report=term-missing
```

The synthetic default is fully offline. The optional real-data path
(`--data-source-pref fred+polygon`) lazily fetches the keyless FRED rate series
and the Polygon PIT cross-section, degrading to the synthetic/committed panel on
any failure — it never hard-fails.

## Limitations

These are first-order threats to the causal interpretation, not footnotes. They
are why the verdict is conservative by construction.

- **Event-study confounding (identification, not just noise).** An event study
  measures association in a window, not a clean exogenous shock. Other macro
  releases (CPI, NFP, Treasury auctions) cluster near FOMC dates, and the
  announcement itself bundles the rate decision with the statement, the dot plot,
  and the press conference. CAR attributes the entire windowed move to "the
  announcement"; it cannot isolate the marginal causal effect of the rate
  decision alone. A significant CAR is consistent with, but does not prove, a
  causal Fed effect.
- **FOMC surprise proxy is coarse.** The surprise sign is the realized
  target-rate change observable at the announcement, not the *unexpected*
  component. Markets price expectations, so the policy-relevant shock is the
  surprise relative to fed-funds-futures (or OIS) — which this tool does **not**
  use. Many "hawkish" meetings were fully anticipated and should carry near-zero
  surprise; classifying by realized sign mislabels them and attenuates / mislabels
  the heterogeneity contrast.
- **Real-data universe is a STATIC basket, not as-of PIT membership.** The
  real-data path scores a fixed modern basket of rate-sensitive (financials + a
  long-duration Treasury proxy) and rate-insensitive large-caps — it does **not**
  resolve point-in-time index membership as-of each event date. It therefore
  carries full survivorship bias (only names that exist *today* are scored), which
  would, if anything, inflate an apparent effect — so the honest null is the
  conservative reading. An as-of PIT resolver over the vendored Polygon
  grouped-daily universe is documented future work; the deployed default runs on
  the synthetic panel, where this does not apply.
- **Clustered-event estimation contamination.** Each event's expected-return model
  is fit on the [t−130, t−11] estimation window, which — when FOMC meetings are
  closely spaced — can overlap *prior* events' event windows, so earlier abnormal
  returns slightly bias a later event's estimated betas. The synthetic default
  spaces events widely enough to bound this, and the no-straddle guard prevents a
  window from crossing its *own* split, but the estimation slice is not purged of
  *other* events' windows. A purge of prior event windows from each estimation
  slice (standard for clustered events) is documented future work.
- **Synthetic default is a machinery test, not market evidence.** The deployed
  default and every committed metric come from a seeded synthetic panel with a
  *known* injected CAR and rate-sensitivity heterogeneity. This validates that the
  placebo/HAC/DiD/multiple-testing stack recovers ground truth and refuses to
  manufacture an effect — it is **not** a claim about live markets. Real-data runs
  require the keyless-FRED + Polygon-PIT path and inherit all of the above.

## Design & decisions

- [`docs/DESIGN.md`](docs/DESIGN.md) — the pipeline, leakage guards, and why the
  verdict is a pure function.
- [`docs/decisions/`](docs/decisions/) — architecture decision records:
  [estimation-window-only market model](docs/decisions/0001-estimation-window-only-market-model.md),
  [placebo as primary significance](docs/decisions/0002-placebo-as-primary-significance.md),
  [multiple-testing correction](docs/decisions/0003-multiple-testing-correction.md),
  [honest heterogeneity, not alpha](docs/decisions/0004-honest-heterogeneity-not-alpha.md),
  [point-in-time universe](docs/decisions/0005-pit-universe.md).

## References

- Brown, S. J., & Warner, J. B. (1985). *Using Daily Stock Returns: The Case of
  Event Studies.* Journal of Financial Economics, 14(1).
- Boehmer, E., Musumeci, J., & Poulsen, A. B. (1991). *Event-Study Methodology
  under Conditions of Event-Induced Variance.* Journal of Financial Economics,
  30(2).
- MacKinlay, A. C. (1997). *Event Studies in Economics and Finance.* Journal of
  Economic Literature, 35(1).
- Romano, J. P., & Wolf, M. (2005). *Exact and Approximate Stepdown Methods for
  Multiple Hypothesis Testing.* Journal of the American Statistical Association,
  100(469).

## License

MIT — see [LICENSE](LICENSE).
