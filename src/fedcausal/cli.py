"""Command-line interface (Typer).

A thin orchestration layer over the compute library: load the event panel, run
the event study / placebo / DiD, and print the summary. Typer is built on the
standard library, but constructing the app object is deferred to :func:`build_app`
so importing this module has no side effects (no command registration or I/O at
import time). The module-level ``app`` is a lazily-built singleton consumed by the
``fedcausal`` console-script entry point.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``eventstudy``, ``placebo``, ``did``) on a fresh
    ``typer.Typer`` instance. Typer is imported lazily inside this function so
    that importing :mod:`fedcausal.cli` does not import Typer or register any
    commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    raise NotImplementedError


def eventstudy(
    event_window: int = 1,
    estimation_window: int = 120,
    model: str = "market",
    surprise: str = "all",
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> int:
    """Run the event study (abnormal returns + CAR + HAC) and print the summary.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k`` (window ``[-k, +k]``).
    estimation_window:
        The pre-event estimation-window length.
    model:
        The expected-return model (``"market"`` or ``"mean_adjusted"``).
    surprise:
        The surprise subset (``"all"``, ``"hawkish"`` or ``"dovish"``).
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    raise NotImplementedError


def placebo(
    event_window: int = 1,
    estimation_window: int = 120,
    n_placebo: int = 500,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> int:
    """Run placebo-date randomization and print the observed-CAR percentile.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k``.
    estimation_window:
        The pre-event estimation-window length.
    n_placebo:
        Number of placebo-date draws.
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    raise NotImplementedError


def did(
    event_window: int = 1,
    estimation_window: int = 120,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> int:
    """Run the rate-sensitivity difference-in-differences and print the coefficient.

    Parameters
    ----------
    event_window:
        The event-window half-width ``k``.
    estimation_window:
        The pre-event estimation-window length.
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    raise NotImplementedError
