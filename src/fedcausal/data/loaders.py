"""Real-data loaders: keyless FRED rate series + Polygon PIT single-name returns.

The synthetic panel (:mod:`fedcausal.data.synthetic`) is the default everywhere.
These loaders provide the OPTIONAL real-data path described in the brief:

- **FRED is free and keyless** via the ``fredgraph`` CSV endpoint
  (``https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU`` for the
  target-rate upper bound, ``DFF`` for the effective rate). It is fetched lazily
  with ``httpx`` to classify each meeting's surprise sign, degrading to the
  committed snapshot in :mod:`fedcausal.events.calendar_data` on any failure.
- **Polygon** supplies the single-name cross-section via the existing
  point-in-time (PIT) universe + price provider
  (:mod:`fedcausal.data_providers.polygon`); a ticker is only included if it was
  in the index as-of the event (PIT, no survivorship leak).

LEAKAGE GUARDS: simple returns are computed with ``pct_change(fill_method=None)``
(prices are never forward-filled before differencing); the FRED series is read at
its RELEASE date so the surprise uses only information available at the
announcement.

LAZY IMPORTS: ``httpx`` and the Polygon provider are imported INSIDE the
functions, never at module import time. Importing this module has no side
effects (no network).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from fedcausal._typing import DataSource
    from fedcausal.data.synthetic import SyntheticPanel
    from fedcausal.events.calendar import FOMCEvent

#: Keyless FRED ``fredgraph`` CSV endpoint template (no API key required).
FREDGRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

#: FRED series ids used by the loaders.
FRED_TARGET_UPPER = "DFEDTARU"  # federal funds target range, upper bound
FRED_EFFECTIVE = "DFF"  # daily effective federal funds rate


def fetch_fred_series(
    series_id: str,
    *,
    start: date | None = None,
    end: date | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch a FRED series from the keyless ``fredgraph`` CSV endpoint (lazy httpx).

    LAZY IMPORT: ``httpx`` (the ``data`` extra) is imported inside this function;
    importing the module touches no network. The CSV is parsed into a date-indexed
    float series, with FRED's ``.`` missing markers dropped.

    Parameters
    ----------
    series_id:
        The FRED series id (e.g. ``"DFEDTARU"`` or ``"DFF"``).
    start, end:
        Optional inclusive date bounds applied after download.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    pandas.Series
        The date-indexed series (percent), sorted ascending.

    Raises
    ------
    ValidationError
        If ``series_id`` is empty.
    RuntimeError
        On network/parse failure (callers decide whether to fall back).
    """
    raise NotImplementedError


def load_surprise_labels(
    events: list[FOMCEvent],
    *,
    use_fred: bool = False,
) -> list[FOMCEvent]:
    """Refresh each event's surprise label from live FRED, else keep committed.

    With ``use_fred=True`` this fetches ``DFEDTARU`` and re-derives the signed
    rate change per meeting from the series read at the announcement's RELEASE
    date (no future revision); on any failure it returns the committed
    ``events`` unchanged. With ``use_fred=False`` it is a pass-through.

    Parameters
    ----------
    events:
        The committed FOMC events to (optionally) refresh.
    use_fred:
        Whether to attempt a live FRED refresh.

    Returns
    -------
    list[FOMCEvent]
        Events with surprise labels (refreshed or committed).
    """
    raise NotImplementedError


def fetch_polygon_returns(
    tickers: list[str],
    start: date,
    end: date,
    *,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch single-name simple returns from Polygon (lazy provider import).

    LAZY IMPORT: :class:`fedcausal.data_providers.polygon.PolygonProvider` (and,
    inside it, ``httpx``) is imported here, never at module import time. Adjusted
    closes are converted to simple returns with ``pct_change(fill_method=None)``
    (no forward-fill before differencing).

    Parameters
    ----------
    tickers:
        PIT-resolved index members to fetch.
    start, end:
        Inclusive date range.
    api_key:
        Optional explicit Polygon key (else resolved from env/.env).

    Returns
    -------
    pandas.DataFrame
        Wide panel of single-name simple returns (rows = date, columns = ticker).

    Raises
    ------
    ValidationError
        If ``tickers`` is empty or ``end <= start``.
    """
    raise NotImplementedError


def load_event_panel(
    *,
    data_source_pref: DataSource = "synthetic",
    seed: int = 7,
    use_fred: bool = False,
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> tuple[SyntheticPanel, DataSource]:
    """Load an event panel, defaulting to synthetic, with a real-data option.

    Resolution: ``data_source_pref="synthetic"`` (the default) returns the seeded
    synthetic panel directly. ``"fred+polygon"`` attempts to assemble a real
    panel (FRED surprise labels + Polygon PIT single-name returns) and DEGRADES
    to the synthetic panel on any failure, so the loader NEVER hard-fails.

    Parameters
    ----------
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed for the synthetic panel / fallback.
    use_fred:
        Whether to refresh surprise labels from live FRED.
    tickers:
        Optional explicit PIT ticker list for the Polygon path.
    start, end:
        Optional inclusive date range for the real-data path.

    Returns
    -------
    tuple[SyntheticPanel, DataSource]
        The panel and the source it actually came from (``"synthetic"`` on
        fallback).
    """
    raise NotImplementedError
