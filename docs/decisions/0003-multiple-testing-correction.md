# ADR-0003 — Multiple-testing correction over the full spec grid

**Status:** Accepted

## Context

The event study is not a single test. There is a grid of reasonable
specifications: event-window half-widths (`k`), expected-return models (market vs
mean-adjusted), and surprise subsets (all / hawkish / dovish). If a researcher is
free to scan that grid and report the most significant cell, the per-comparison
_p_-value is meaningless — with enough specifications, *something* clears 0.05 by
chance. This is the garden of forking paths, and it is how fragile "Fed effects"
get published.

An honest significance claim must pay for every test it could have run, not only the
one it chose to report.

## Decision

Apply a **multiple-testing correction across the full specification grid** and report
an **honest `n_tests`** count (the number of cells in windows × models × surprise
subsets, not a single favorable cell). Two complementary corrections are provided:

- **Benjamini-Hochberg** for false-discovery-rate control, matched to statsmodels;
- **Romano-Wolf** stepdown for family-wise-error-rate (FWER) control, which accounts
  for dependence between the specifications.

Leg (3) of `fed_effect_is_tradable` requires that **at least one specification
survives** the correction across the full grid — not that an uncorrected cell is
significant.

Enforced by:

- `parity/test_parity_multiple_testing.py::test_benjamini_hochberg_matches_statsmodels`
  and `::test_benjamini_hochberg_adjusted_are_monotone_in_raw_order`;
- `parity/test_parity_multiple_testing.py::test_romano_wolf_matches_hand_reference`
  and `::test_romano_wolf_controls_fwer_on_global_null` (FWER control on the global
  null — the correction does not over-reject when nothing is true).

## Consequences

- **Positive.** The reported significance is honest about the search space. The FWER
  test on the global null proves the correction does not manufacture rejections.
  `n_tests` is surfaced in the summary so a reader can see the denominator.
- **Negative / accepted.** Corrected significance is conservative; a genuine but
  marginal effect in one cell may not survive. For an honest-null deliverable this is
  the *correct* bias — we would rather miss a fragile effect than report a spurious
  one. BH and Romano-Wolf are both provided because FDR and FWER answer different
  questions; the verdict uses the stricter survival criterion.
- Romano-Wolf requires the joint null distribution (via the placebo resampling),
  which ties this decision to ADR-0002.
