# ADR-0002 — Placebo-date randomization is the primary significance source

**Status:** Accepted

## Context

The natural significance test for a mean CAR is a cross-sectional _t_-statistic. On
the deployed-default panel that _t_-stat is ~5.5 — large in isolation. But a
parametric _t_-stat rests on assumptions that event studies routinely violate:
returns are cross-sectionally correlated (especially on macro days when everything
moves together), event-induced variance inflates the denominator, and clustering of
events in calendar time induces serial dependence. A raw _t_-stat can therefore look
decisive while measuring nothing more than market-wide comovement on busy days.

A credible significance statement needs a null that is built from the *same data
generating process* minus the event — i.e. an empirical null that inherits the same
cross-sectional correlation and volatility structure.

## Decision

Make **placebo-date randomization the primary significance source.** Re-run the
entire CAR computation on a large number (`n_placebo`, default 500) of random
**non-event** dates, drawn so they **exclude every real event window** (no
contamination), and form the null distribution of the placebo CARs. The observed CAR
is significant only if it lands in the tail of that placebo distribution
(`placebo_pctile` / `placebo_pvalue`).

A raw cross-sectional _t_-stat **alone is never sufficient** for the verdict. The
placebo percentile is leg (1) of the four-part `fed_effect_is_tradable` test; the
HAC-robust _t_ is a *separate* leg (2), not a substitute.

Enforced by:

- `property/test_eventstudy_invariants.py::test_placebo_dates_exclude_every_real_event_window`
  (no contamination);
- `regression/test_regression_honest_null.py::test_placebo_percentile_centered_under_no_effect`
  and `regression/test_regression_eventstudy.py::test_placebo_percentile_uniform_under_no_effect`
  (the percentile is ~uniform when there is no real effect, so the test is calibrated).

## Consequences

- **Positive.** The null inherits the real cross-sectional correlation and volatility
  structure, so the significance statement is robust to the comovement that breaks a
  parametric _t_-stat. The uniformity-under-no-effect test proves the placebo test is
  calibrated rather than rigged.
- **Negative / accepted.** Placebo randomization is computationally heavier than a
  closed-form _t_-stat (it re-runs the event study `n_placebo` times); `n_placebo` is
  capped on the hosted path to bound cost. The placebo draw uses the same seeded RNG,
  so results are reproducible.
- Because the placebo null is the primary source, a significant CAR that fails the
  placebo test is correctly reported as *not* significant — which is exactly the kind
  of false positive the honest-null deliverable is built to catch.
