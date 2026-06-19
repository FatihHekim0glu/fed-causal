# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Implemented `serve.run_analysis` — the single hosted entrypoint that wires every
  module group into one leakage-free pipeline (load panel → estimation-window-only
  abnormal returns + CAR → cross-sectional t / BMP / HAC → placebo-date null →
  rate-sensitivity DiD + net-of-cost spread → Benjamini-Hochberg correction → the
  PURE `fed_effect_is_tradable` verdict), returning a JSON-serializable summary
  (all scalars via `_safe_float`), a CAR-path figure with a confidence band, a
  placebo null-distribution figure, and a `RunManifest`. No network on the default
  path; the `fred+polygon` preference degrades to synthetic offline.
- A genuine `surprise` subset filter so the event-study/placebo battery narrows to
  the requested surprise sign (the DiD always contrasts hawkish vs. dovish).
- Committed deployed-default reference (`src/fedcausal/artifacts/reference.json`)
  with a `load_reference()` loader and a reproducible `scripts/build_reference.py`
  generator (`--check` drift guard, wired into CI). The artifact holds the
  deployed-default summary, the known-CAR-recovery numbers, and the pure-noise
  honest-null numbers.
- Integration tests (end-to-end synthetic → windows → abnormal/CAR → placebo + HAC
  + DiD → verdict, no network; figures are `{data, layout}` dicts), the honest-null
  regression (`pure_noise → fed_effect_is_tradable=False`, deterministic across
  `PYTHONHASHSEED`), the known-CAR recovery regression, a reference-artifact parity
  regression, and unit coverage for the `serve` validation/surprise-filter helpers.
- Packaged `reference.json` + `py.typed` into the wheel.

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
