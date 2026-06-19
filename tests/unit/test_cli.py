"""Unit tests for the Typer CLI (:mod:`fedcausal.cli`).

The CLI wires the compute kernels together itself (it does NOT call the still-
stubbed :func:`fedcausal.serve.run_analysis`). These tests assert:

- ``run_pipeline`` runs the full leakage-free stack on the seeded synthetic panel
  and returns the PURE ``fed_effect_is_tradable=False`` verdict (honest-null);
- the ``eventstudy`` / ``placebo`` / ``did`` command functions print their block,
  surface the "Tradable Fed effect: NO" verdict, and exit ``0``;
- ``build_app`` constructs a Typer app exposing exactly the three commands; and
- importing :mod:`fedcausal.cli` does NOT import Typer (it is lazily imported).

Placebo draws are kept small (``n_placebo`` low) so the suite stays fast.
"""

from __future__ import annotations

import pytest

from fedcausal.cli import (
    PipelineSummary,
    build_app,
    did,
    eventstudy,
    placebo,
    run_pipeline,
)

pytestmark = pytest.mark.unit

#: Small placebo count keeps the deterministic pipeline fast in the unit suite.
_FAST_PLACEBO = 64


def test_run_pipeline_synthetic_is_not_tradable() -> None:
    """The seeded synthetic default yields an honest-null ``False`` verdict."""
    summary = run_pipeline(n_placebo=_FAST_PLACEBO, seed=7)

    assert isinstance(summary, PipelineSummary)
    assert summary.data_source == "synthetic"
    assert summary.fed_effect_is_tradable is False
    # A real injected CAR is present, so the event-study stats are non-trivial...
    assert summary.n_events >= 8
    assert summary.car_mean != 0.0
    # ...but the honest-null verdict still reads False (the DiD spread does not
    # clear costs / multiple-testing the way a tradable alpha would).
    assert summary.n_tests >= 2


def test_run_pipeline_is_deterministic() -> None:
    """The same seed reproduces the same summary byte-for-byte."""
    a = run_pipeline(n_placebo=_FAST_PLACEBO, seed=7)
    b = run_pipeline(n_placebo=_FAST_PLACEBO, seed=7)
    assert a.to_dict() == b.to_dict()


def test_run_pipeline_mean_adjusted_model() -> None:
    """The ``mean_adjusted`` model path also runs and stays an honest-null."""
    summary = run_pipeline(n_placebo=_FAST_PLACEBO, model="mean_adjusted", seed=7)
    assert summary.fed_effect_is_tradable is False


def test_eventstudy_command_prints_verdict(capsys: pytest.CaptureFixture[str]) -> None:
    """``eventstudy`` prints the event-study block and the NO verdict, exit 0."""
    # Monkeypatch-free: the default n_placebo (500) is fine but slower; call the
    # underlying pipeline path through the public function with defaults shrunk by
    # the command's own default. We accept the default here to exercise the real
    # command, but keep the assertion on the printed verdict.
    code = eventstudy(seed=7)
    out = capsys.readouterr().out

    assert code == 0
    assert "fed-causal event study" in out
    assert "Tradable Fed effect: NO" in out
    assert "data source        : synthetic" in out
    assert "placebo percentile" in out


def test_placebo_command_prints_percentile(capsys: pytest.CaptureFixture[str]) -> None:
    """``placebo`` prints the placebo block (with the percentile) and exits 0."""
    code = placebo(n_placebo=_FAST_PLACEBO, seed=7)
    out = capsys.readouterr().out

    assert code == 0
    assert "placebo-date randomization" in out
    assert "placebo percentile" in out
    assert "Tradable Fed effect: NO" in out


def test_did_command_prints_coefficient(capsys: pytest.CaptureFixture[str]) -> None:
    """``did`` prints the DiD coefficient block and exits 0."""
    code = did(seed=7)
    out = capsys.readouterr().out

    assert code == 0
    assert "difference-in-differences" in out
    assert "DiD coefficient" in out
    assert "Tradable Fed effect: NO" in out


def test_command_returns_error_code_on_bad_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A library ``FedCausalError`` (bad geometry) is caught and returns exit 1."""
    # estimation_window=1 is below the kernel's minimum (>= 2) -> ValidationError,
    # which is a FedCausalError the command catches and turns into exit code 1.
    code = eventstudy(estimation_window=1, seed=7)
    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("error:")


def test_build_app_registers_three_commands() -> None:
    """``build_app`` returns a Typer app exposing eventstudy/placebo/did."""
    app = build_app()
    # Typer registers commands as ``CommandInfo`` objects; their callbacks carry
    # the command name (the explicit string passed to ``@cli.command``).
    names = {info.name for info in app.registered_commands}
    assert names == {"eventstudy", "placebo", "did"}


def test_build_app_runs_via_runner() -> None:
    """The Typer app dispatches a real ``did`` invocation end-to-end (exit 0)."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(build_app(), ["did", "--seed", "7"])
    assert result.exit_code == 0
    assert "Tradable Fed effect: NO" in result.stdout


def test_importing_cli_does_not_import_typer() -> None:
    """Importing :mod:`fedcausal.cli` in a fresh interpreter pulls in no Typer."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import fedcausal.cli\n"
        "assert 'typer' not in sys.modules, 'typer leaked on import'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
