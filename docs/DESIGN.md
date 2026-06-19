# fed-causal — design

This document is the source-of-truth narrative for *why* `fed-causal` is built the
way it is. The README is the user-facing front door; this is the engineering and
econometrics rationale. Architecture-level decisions that have trade-offs are
recorded as ADRs under [`docs/decisions/`](decisions/).

## The deliverable

`fed-causal` is an **honest-null** tool. The headline result is a *negative*: around
FOMC announcements, cumulative abnormal returns (CAR) show a transient average move,
but after placebo-date randomization, HAC standard errors, and a multiple-testing
correction, the move is **cross-sectional heterogeneity** (rate-sensitive names move
more, mechanically) — **not** a placebo-robust, tradable cross-sectional alpha.

The value of the project is therefore the *rigor* of the pipeline, not a profit
claim. Everything in the design exists to make the null **earned** rather than
assumed: a real effect must clear four independent bars before the verdict flips,
and the synthetic ground-truth tests prove the machinery can both recover a planted
effect and refuse to invent one.

## Pipeline (single leakage-free path)

`serve.run_analysis` is the one hosted entrypoint and wires every module group into
one ordered, leakage-free pipeline:

```
load panel (synthetic default | fred+polygon)
  → events: classify surprise (sign available at announcement) + build windows
  → eventstudy.abnormal: market model fit on the ESTIMATION WINDOW ONLY → CAR
  → eventstudy.tests: cross-sectional t · BMP · HAC / clustered SE
  → eventstudy.placebo: re-run CAR on random NON-event dates → null distribution
  → did.model: rate-sensitive treated vs control, clustered SE → net-of-cost spread
  → evaluation.multiple_testing: BH / Romano-Wolf across the full spec grid
  → evaluation.verdict: PURE fed_effect_is_tradable
```

The pipeline returns a JSON-serializable summary (all scalars via `_safe_float`), a
CAR-path figure with a confidence band, a placebo null-distribution figure, and a
`RunManifest` with a BLAKE2b config hash for reproducibility.

## Module map

| Module group | Responsibility |
| --- | --- |
| `data/synthetic.py` | Seeded event panel: known injected CAR + rate-sensitivity heterogeneity + a pure-noise variant. The ground truth the test suite validates against. |
| `data/loaders.py` | Keyless FRED rate series (release-date aware) + Polygon PIT single-name returns + the FOMC calendar. Lazy `httpx`; synthetic default. |
| `events/calendar.py` | Committed FOMC announcement dates + hawkish/dovish/neutral classification from the rate-change **sign available at the announcement**. |
| `events/windows.py` | Estimation window `[t-130, t-11]` vs event window `[-k, +k]`; no overlap, no straddle, deterministic. |
| `eventstudy/abnormal.py` | Expected returns from a market (or mean-adjusted) model fit on the **estimation window only**; abnormal = actual − expected; CAR over the event window. |
| `eventstudy/tests.py` | Cross-sectional _t_, Boehmer-Musumeci-Poulsen standardized-residual statistic, HAC / clustered SEs. |
| `eventstudy/placebo.py` | Placebo-date randomization: CAR on random non-event dates (excluding all real event windows) → the **primary** significance source. |
| `did/model.py` | Difference-in-differences: rate-sensitive treated vs control around hawkish vs dovish surprises, clustered SEs. |
| `did/heterogeneity.py` | The honest cross-sectional-heterogeneity framing; SCM demoted to descriptive. |
| `evaluation/multiple_testing.py` | BH / Romano-Wolf across the window × model × surprise grid, with an honest `n_tests`. |
| `evaluation/verdict.py` | The **pure** `fed_effect_is_tradable`. |
| `plots.py`, `cli.py`, `serve.py` | Lazy Plotly figures, the Typer CLI (`eventstudy`/`placebo`/`did`), and the hosted entrypoint. |

## Leakage & correctness guards

These are the invariants that make the null *honest*. Each is enforced in code and
pinned by a property or regression test (see the README correctness-gates table):

1. **Estimation-window-only fit.** The expected-return model is fit on the pre-event
   estimation window only; a property test perturbs event-window returns and asserts
   the fitted betas are byte-identical (ADR-0001).
2. **No window overlap / straddle.** Estimation and event windows never overlap; a
   property test enumerates configurations and checks non-overlap and non-straddle.
3. **Placebo excludes real events.** Placebo dates are drawn from non-event dates and
   exclude every real event window — no contamination (ADR-0002).
4. **Announcement-time information only.** The event timestamp is the announcement
   *date* (no pre-announcement signal); surprise classification uses only the rate
   information available at the announcement (no future revisions).
5. **Point-in-time universe.** The cross-section uses the issuer's membership as-of
   the event (ADR-0005), and returns use `pct_change(fill_method=None)`.
6. **HAC / clustered SEs.** Standard errors are HAC (serial correlation) and
   clustered (cross-sectional), for both the event study and the DiD (ADR-0003).
7. **Honest multiple-testing denominator.** The correction counts the *full* spec
   grid (windows × models × surprise subsets), not a single favorable cell.

## The verdict is a pure function

`fed_effect_is_tradable` is computed, never narrated. It is `False` unless **all
four** hold:

1. the observed CAR is significant against the **placebo-date null** (the primary
   significance source — never a raw _t_-stat alone), **and**
2. the mean CAR is significant under a **HAC / Newey-West** SE, **and**
3. at least one specification **survives the multiple-testing correction**, **and**
4. the rate-sensitivity DiD implies a long/short spread that is positive and
   significant **net of transaction costs**.

On the deployed default the first three legs clear and the fourth fails (the
net-of-cost DiD spread is not significant), so the pure verdict is `false` — that is
the deliverable (ADR-0004). The pure-noise control fails every leg.

## Non-goals

- **No torch / ONNX / sklearn.** This is a pure numpy/scipy/statsmodels
  econometrics tool; the event-study OLS on the estimation window is cheap and runs
  per request without refitting any heavy model.
- **No profit claim.** SCM/DiD are descriptive heterogeneity, never alpha.
- **No import-time I/O.** `src/fedcausal/` has zero import-time side effects; all
  network clients are lazy and all demos sit behind `__main__`.

## References

- Brown, S. J., & Warner, J. B. (1985). *Using Daily Stock Returns: The Case of
  Event Studies.* Journal of Financial Economics, 14(1).
- Boehmer, E., Musumeci, J., & Poulsen, A. B. (1991). *Event-Study Methodology under
  Conditions of Event-Induced Variance.* Journal of Financial Economics, 30(2).
- MacKinlay, A. C. (1997). *Event Studies in Economics and Finance.* Journal of
  Economic Literature, 35(1).
- Romano, J. P., & Wolf, M. (2005). *Exact and Approximate Stepdown Methods for
  Multiple Hypothesis Testing.* Journal of the American Statistical Association,
  100(469).
