# Contributing

Thanks for your interest in `fed-causal`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with the lean extras + dev tooling.
uv venv
uv pip install -e ".[data,viz,cli,dev]"
```

This is a **torch-free** econometrics tool: the install is intentionally lean —
`numpy/scipy/statsmodels` for the core, plus `httpx/pyarrow/diskcache` (real-data
loaders), `plotly/kaleido` (figures), and `typer` (CLI). There is **no** `[all]`
extra and **no** `torch`/`onnx`/`onnxruntime`/`sklearn` anywhere.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src tests                                            # lint
uv run mypy src                                                        # types (strict)
uv run pytest -q --cov=fedcausal --cov-report=term --cov-fail-under=85 # tests + coverage
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) is run on every PR. It is currently non-blocking in
  CI while residual strict-mode issues are burned down, but new code should not
  add type errors.
- **Tests** (`pytest`) must pass with **coverage ≥ 85%** (the gate also lives in
  `[tool.coverage.report] fail_under` in `pyproject.toml`).

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## Correctness invariants (do not regress)

This repo's value is its leakage discipline. New code must keep:

- the expected-return model fit on the **estimation window only** (perturbing
  event-window returns must leave the fitted betas byte-identical);
- estimation and event windows **never overlapping or straddling**;
- placebo dates **excluding every real event window**;
- the **placebo null** as the PRIMARY significance source (a raw t-stat alone is
  never sufficient for the verdict);
- `fed_effect_is_tradable` a **pure function** that reads `False` unless the
  effect is placebo-significant AND HAC-robust AND survives multiple testing AND
  yields a net-of-cost tradable DiD spread.

## Import purity

`src/fedcausal/` must have **zero import-time side effects** — no network, no
FRED/Polygon calls, and no `statsmodels` import at module load. Heavy/optional
dependencies are imported lazily inside the functions that need them.

## Commit hygiene

- Use clear, present-tense commit messages.
- **Do not** add AI-attribution trailers — no `Co-Authored-By: Claude`,
  no "Generated with Claude", no robot-emoji attribution lines. The
  `.github/workflows/no-ai-attribution.yml` guard fails any PR that contains them.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
