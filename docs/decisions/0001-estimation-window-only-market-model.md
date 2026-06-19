# ADR-0001 — Estimation-window-only market model

**Status:** Accepted

## Context

An event study needs a counterfactual: what return would each name have realized
*absent* the event? The expected-return model (here a market model, optionally a
mean-adjusted model) supplies that counterfactual, and the abnormal return is
`actual − expected`. The cumulative abnormal return (CAR) over the event window is
the headline statistic.

The single most damaging mistake in event-study methodology is fitting the
expected-return model on data that includes the event window. If the event-window
returns inform the betas, the "expected" return absorbs part of the event itself,
the abnormal return is mechanically shrunk toward zero, and the standard errors are
contaminated by event-induced variance. This is a leakage bug dressed as a
modelling choice.

## Decision

Fit the expected-return model on the **pre-event estimation window only**
(`[t-130, t-11]` by default), strictly disjoint from the event window `[-k, +k]`.
The estimation and event windows never overlap and never straddle. The fitted betas
are a function of pre-event data alone.

This is enforced, not merely intended:

- a property test enumerates window configurations and asserts non-overlap and
  non-straddle (`property/test_events_windows.py`);
- a property test perturbs the event-window returns arbitrarily and asserts the
  fitted market-model betas are **byte-identical**
  (`property/test_eventstudy_invariants.py::test_market_model_beta_invariant_to_event_window_perturbation`).

The market-model abnormal returns are also pinned against a statsmodels OLS
reference (`parity/test_parity_eventstudy.py`).

## Consequences

- **Positive.** The counterfactual is leakage-free by construction; the beta
  invariance test makes any future regression that reintroduces event-window data
  fail loudly. Abnormal returns and CAR are comparable to the Brown-Warner / MacKinlay
  literature.
- **Negative / accepted.** A pre-event window assumes the market-model relationship
  is stable into the event window; a structural break right before the event would
  bias the counterfactual. We accept this standard event-study assumption and surface
  confounding as an explicit limitation rather than modelling time-varying betas,
  which would add complexity without changing the honest-null conclusion.
- The estimation-window OLS is cheap, so the hosted backend can fit it per request
  without refitting any heavy model.
