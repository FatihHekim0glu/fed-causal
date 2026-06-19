# ADR-0004 — Honest heterogeneity, not alpha — the pure verdict

**Status:** Accepted

## Context

It is tempting to read a significant CAR + a significant DiD coefficient as "the Fed
announcement is a tradable signal." That conclusion does not follow. The DiD shows
that rate-sensitive ("treated") names move *more* than control names around
hawkish/dovish surprises — but that is **cross-sectional heterogeneity**: a
mechanical, descriptive fact about which names are more rate-exposed, not a profit
opportunity. Turning heterogeneity into alpha requires a long/short spread that is
positive **and** significant **net of transaction costs** — and that is a separate,
stricter bar.

The risk is narrative: a human author could look at four numbers and *talk* their
way to "tradable." The design has to prevent that.

## Decision

Frame DiD/SCM as **descriptive heterogeneity, never alpha**, and make the trading
verdict a **pure function** that is *computed, not narrated*. `fed_effect_is_tradable`
returns `False` unless **all four** legs hold:

1. placebo-date significance (ADR-0002), **and**
2. HAC-robust mean CAR (ADR-0003 SEs / `evaluation/hac.py`), **and**
3. at least one specification survives multiple testing (ADR-0003), **and**
4. the rate-sensitivity DiD implies a long/short spread that is positive **and**
   significant **net of transaction costs** (`did_net_spread`).

On the deployed default, legs 1-3 clear (placebo percentile 100, HAC _p_ = 6.4e-14, a
surviving spec) and leg 4 fails (the net-of-cost DiD spread is not significant), so
the verdict is **derived** `false`. The pure-noise control fails every leg.

Enforced by:

- `unit/test_verdict.py` (the verdict is a pure function of its inputs);
- `regression/test_regression_honest_null.py::test_pure_noise_panel_is_not_tradable`
  and `::test_pure_noise_verdict_deterministic_across_pythonhashseed`;
- `regression/test_regression_honest_null.py::test_reference_artifact_matches_live_run_analysis`
  (the committed `reference.json` verdict matches the live pipeline).

## Consequences

- **Positive.** The verdict cannot be argued up to `true`; it is a deterministic
  function of four measured quantities, reproducible across `PYTHONHASHSEED`. SCM is
  demoted to descriptive so it cannot be mistaken for a backtest. The committed
  reference keeps the published number honest.
- **Negative / accepted.** A genuinely tradable Fed effect that cleared all four legs
  would flip the verdict to `true` — by design, the tool does not hard-code `false`.
  The net-of-cost leg depends on a transaction-cost assumption; we use a conservative
  cost and surface the spread so the assumption is inspectable.
- This ADR is the reason the project exists: the deliverable is the rigorous,
  leakage-free *pipeline that earns the null*, not a profit claim.
