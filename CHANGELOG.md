# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-19

### Added

- Initial package skeleton (src-layout, import name `fedcausal`, `py.typed`).
- Core helpers reused from `hrp-portfolio`: `_constants`, `_typing`,
  `_exceptions` (reframed with a `FedCausalError` base + `WindowOverlapError` /
  `EventCalendarError` for the event-study domain), `_validation`, `_manifest`
  (`RunManifest` with BLAKE2b config-hash), and `_rng` (seeded PCG64 generator +
  substream spawning).
- Vendored real-data plumbing: `data_providers/polygon.py` (Polygon PIT price
  provider, lazy `httpx`) and `evaluation/hac.py` (Newey-West HAC SEs, copied
  algorithm-identical from `pairs-trading`).
- A committed reference FOMC calendar (`events/calendar_data.py`): public
  announcement dates + a target-rate snapshot for surprise classification.
- Typed stubs (signatures + docstrings + frozen/slotted dataclasses with
  `to_dict`) for every module-map module:
  - `data/{synthetic,loaders}` — seeded synthetic event panel (known injected
    CAR + rate-sensitivity heterogeneity + pure-noise variant) and the keyless
    FRED + Polygon PIT loaders.
  - `events/{calendar,windows}` — surprise classification and leakage-safe
    estimation/event windowing.
  - `eventstudy/{abnormal,tests,placebo}` — estimation-window-only market model,
    CAR, cross-sectional t / BMP / HAC tests, and placebo-date randomization.
  - `did/{model,heterogeneity}` — clustered difference-in-differences and the
    honest cross-sectional-heterogeneity framing.
  - `evaluation/{multiple_testing,verdict}` — Benjamini-Hochberg / Romano-Wolf
    correction over the full spec grid and the PURE `fed_effect_is_tradable`
    verdict.
  - `plots.py` (lazy Plotly), `cli.py` (Typer), and a `serve.py` `run_analysis`
    entrypoint.
- Curated, import-pure top-level `__init__.py`.
- Seeded synthetic test fixtures (`synthetic_event_panel`, `rate_sensitive_panel`,
  `pure_noise`) and an import-purity smoke test; partitioned `tests/`
  (unit/parity/property/regression/integration).
- CI (lean extras, py3.11-3.13, non-blocking strict mypy, coverage ≥ 85%) and a
  `no-ai-attribution` guard.

[Unreleased]: https://github.com/FatihHekim0glu/fed-causal/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/fed-causal/releases/tag/v0.1.0
