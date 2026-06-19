# ADR-0005 — Point-in-time universe for the cross-section

**Status:** Accepted

## Context

The single-name cross-section needs a set of issuers to study around each FOMC
event. The naive approach is to take *today's* index constituents and pull their
histories back through every past event. That is survivorship bias: today's members
are the survivors — names that were delisted, acquired, or dropped from the index
are silently excluded, and the surviving names are exactly the ones that performed
well enough to stay. Any effect estimated on a forward-looking universe is biased
upward, because the universe itself encodes the outcome.

For an honest-null tool this is fatal: survivorship bias would *inflate* an apparent
Fed effect and undermine the entire claim.

## Decision

Use a **point-in-time (PIT) universe**: each event uses the issuers that were index
members **as-of that event date**, via the vendored Polygon PIT universe
(`api/lib/polygon/sp500_universe.py`) and the Polygon PIT price provider. Returns are
computed with `pct_change(fill_method=None)` so missing observations are not silently
forward-filled into spurious zero returns.

The synthetic default panel is constructed with a fixed, ground-truth membership so
the machinery is validated without any survivorship question; the PIT universe
applies on the real-data (`fred+polygon`) path.

## Consequences

- **Positive.** The cross-section reflects what an investor could actually have held
  at each event, removing the upward survivorship bias. This makes the honest null
  *conservative*: if anything, PIT membership lowers the apparent effect, so a `false`
  verdict is the safe reading. `pct_change(fill_method=None)` avoids fabricated
  returns from gaps.
- **Negative / accepted.** PIT reconstruction is approximate — the vendored universe
  cannot perfectly capture every historical addition, deletion, ticker change, or
  M&A event, so residual survivorship bias may remain. This is surfaced as an explicit
  README limitation. Perfect historical constituent data is out of scope and would not
  change the honest-null conclusion.
- The real-data path depends on the existing Polygon key and degrades to the
  synthetic/committed panel on any failure, so a PIT-universe lookup failure never
  hard-fails a request.
