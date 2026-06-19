"""Import-purity smoke test: ``import fedcausal`` must be side-effect-free.

The package contract is ZERO import-time side effects: importing ``fedcausal`` (or
any of its submodules) must NOT import a network client (``httpx``), the heavy
econometrics backend (``statsmodels``), the plotting stack (``plotly``), or the
CLI framework (``typer``), and must NOT touch the network. Those live behind lazy
imports inside the functions that need them.

This test runs the import in a FRESH subprocess interpreter so it is unaffected by
modules already imported by the rest of the suite, then asserts the forbidden
modules are absent from ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Modules that must NOT be pulled in merely by importing ``fedcausal``.
FORBIDDEN_ON_IMPORT = (
    "httpx",
    "statsmodels",
    "plotly",
    "typer",
    "kaleido",
    "diskcache",
    # Torch-free guarantee: these must never be importable consequences.
    "torch",
    "onnx",
    "onnxruntime",
    "sklearn",
)


@pytest.mark.unit
def test_import_fedcausal_is_side_effect_free() -> None:
    """A fresh interpreter importing ``fedcausal`` pulls in no forbidden module."""
    forbidden = ", ".join(repr(name) for name in FORBIDDEN_ON_IMPORT)
    code = (
        "import sys\n"
        "import fedcausal\n"
        f"forbidden = [{forbidden}]\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, 'fedcausal import leaked: ' + ', '.join(leaked)\n"
        "assert fedcausal.__version__\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"import-purity subprocess failed:\nstdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


@pytest.mark.unit
def test_submodule_imports_are_side_effect_free() -> None:
    """Importing each public submodule pulls in no forbidden network/heavy module."""
    submodules = (
        "fedcausal.data.loaders",
        "fedcausal.data.synthetic",
        "fedcausal.events.calendar",
        "fedcausal.events.windows",
        "fedcausal.eventstudy.abnormal",
        "fedcausal.eventstudy.tests",
        "fedcausal.eventstudy.placebo",
        "fedcausal.did.model",
        "fedcausal.did.heterogeneity",
        "fedcausal.evaluation.multiple_testing",
        "fedcausal.evaluation.verdict",
        "fedcausal.plots",
        "fedcausal.cli",
        "fedcausal.serve",
        "fedcausal.data_providers.polygon",
    )
    imports = "\n".join(f"import {name}" for name in submodules)
    forbidden = ", ".join(repr(name) for name in ("httpx", "statsmodels", "plotly", "typer"))
    code = (
        "import sys\n"
        f"{imports}\n"
        f"forbidden = [{forbidden}]\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "assert not leaked, 'submodule import leaked: ' + ', '.join(leaked)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"submodule import-purity subprocess failed:\nstdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout
