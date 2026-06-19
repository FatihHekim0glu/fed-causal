# fed-causal

> Leakage-free causal inference on FOMC announcements — an event study of
> cumulative abnormal returns, stress-tested with placebo-date randomization,
> HAC/clustered standard errors, a multiple-testing correction, and a
> rate-sensitivity difference-in-differences. **Torch-free** (numpy/scipy/statsmodels).

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](https://github.com/FatihHekim0glu/fed-causal)
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

> _Filled in once the compute kernels land (this is the scaffold commit)._

| Check | What it asserts | Status |
| --- | --- | --- |
| Known-CAR recovery | Injected CAR recovered within tolerance on the synthetic panel | _pending_ |
| Honest-null guard | Pure-noise panel → `fed_effect_is_tradable = False` after placebo + multiple testing | _pending_ |
| Placebo uniformity | Observed-CAR percentile ~uniform on a no-effect panel | _pending_ |
| Estimation/event no-overlap | Windows never overlap or straddle (property test) | _pending_ |
| Beta invariance | Perturbing event-window returns leaves fitted betas byte-identical | _pending_ |
| HAC parity | Newey-West SE matches the reference to 1e-10 | _pending_ |
| BMP / BH / Romano-Wolf parity | Statistics match hand/reference values | _pending_ |

## Limitations

- **Event-study confounding:** other macro releases can cluster near FOMC dates;
  abnormal returns attribute the move to the announcement window, not to a clean
  exogenous shock.
- **FOMC surprise proxy:** the surprise sign is the realized target-rate change,
  a coarse proxy for the *unexpected* component (no fed-funds-futures surprise).
- **PIT survivorship:** the Polygon point-in-time universe approximates index
  membership and cannot perfectly reconstruct historical constituents.
- **Synthetic default:** the deployed default is a seeded synthetic panel; it
  validates the machinery against ground truth but is not real market data.

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
