"""Historical market-data utilities for quantitative research.

This module keeps the data layer explicit and conservative:

- `close` is the vendor-reported raw close for the session.
- `adjusted_close` reflects split/dividend adjustments from the data vendor.
- simple returns are arithmetic percentage changes, `(P_t / P_{t-1}) - 1`.
- log returns are continuously compounded returns, `log(P_t / P_{t-1})`.

For most equity backtests, returns should usually be computed from
`adjusted_close`, because raw `close` does not account for corporate actions.
This module does not forward-fill prices across trading gaps or across symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

CacheFormat = Literal["csv", "parquet"]
PriceLike = pd.Series | pd.DataFrame

STANDARD_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)
YFINANCE_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Volume": "volume",
}


@dataclass(frozen=True)
class DataPaths:
    """Repository paths for raw and processed datasets."""

    raw: Path
    processed: Path


def get_default_data_paths(base_dir: Path | None = None) -> DataPaths:
    """Return repository data directories."""

    root = base_dir or Path(__file__).resolve().parents[2]
    return DataPaths(raw=root / "data" / "raw", processed=root / "data" / "processed")


def download_daily_ohlcv(
    symbols: str | Sequence[str],
    *,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    use_cache: bool = False,
    cache_format: CacheFormat = "csv",
    cache_name: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Download, validate, standardize, and optionally cache daily OHLCV data.

    Parameters
    ----------
    symbols:
        One symbol or a sequence of symbols.
    start, end:
        Optional date bounds passed through to `yfinance.download`.
    use_cache:
        If `True`, load cached processed data when available and save new results
        after cleaning when no cache file exists.
    cache_format:
        Cache file format under `data/processed`.
    cache_name:
        Optional explicit cache stem. By default a deterministic name is derived
        from the symbol set and date bounds.
    data_dir:
        Optional repository root override used mainly for tests.
    """

    normalized_symbols = _normalize_symbols(symbols)
    start_ts = _coerce_bound(start, "start")
    end_ts = _coerce_bound(end, "end")
    _validate_date_bounds(start_ts, end_ts)

    cache_path = _build_cache_path(
        symbols=normalized_symbols,
        start=start_ts,
        end=end_ts,
        use_cache=use_cache,
        cache_format=cache_format,
        cache_name=cache_name,
        data_dir=data_dir,
    )

    if cache_path is not None and cache_path.exists():
        return _load_cached_frames(cache_path, cache_format)

    raw = yf.download(
        tickers=normalized_symbols,
        start=start_ts,
        end=end_ts,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=False,
    )

    if raw.empty:
        joined = ", ".join(normalized_symbols)
        raise ValueError(f"No historical data returned for symbols: {joined}.")

    frames = {
        symbol: _clean_symbol_frame(symbol, _extract_symbol_frame(raw, symbol))
        for symbol in normalized_symbols
    }

    if cache_path is not None:
        _save_cached_frames(frames, cache_path, cache_format)

    return frames


def simple_returns(
    prices: PriceLike,
    *,
    price_column: str | None = None,
) -> PriceLike:
    """Compute arithmetic returns without filling missing observations."""

    series_or_frame = _coerce_price_input(prices, price_column=price_column)
    returns = series_or_frame.pct_change(fill_method=None)
    return returns


def log_returns(
    prices: PriceLike,
    *,
    price_column: str | None = None,
) -> PriceLike:
    """Compute log returns without filling missing observations."""

    series_or_frame = _coerce_price_input(prices, price_column=price_column)
    shifted = series_or_frame.shift(1)
    return np.log(series_or_frame / shifted)


def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    """Normalize symbol input and reject ambiguous cases."""

    raw_symbols = [symbols] if isinstance(symbols, str) else list(symbols)
    cleaned = [symbol.strip().upper() for symbol in raw_symbols if symbol.strip()]

    if not cleaned:
        raise ValueError("At least one non-empty symbol is required.")

    unique = list(dict.fromkeys(cleaned))
    return unique


def _coerce_bound(value: str | date | datetime | None, name: str) -> pd.Timestamp | None:
    """Convert a date-like bound to a normalized timestamp."""

    if value is None:
        return None

    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas controls exact error type
        raise ValueError(f"Invalid {name} date: {value!r}.") from exc

    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _validate_date_bounds(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> None:
    """Reject impossible date bounds."""

    if start is not None and end is not None and start >= end:
        raise ValueError(
            f"Expected start date before end date, received start={start.date()} "
            f"and end={end.date()}."
        )


def _build_cache_path(
    *,
    symbols: Sequence[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    use_cache: bool,
    cache_format: CacheFormat,
    cache_name: str | None,
    data_dir: Path | None,
) -> Path | None:
    """Build the processed-data cache path when caching is enabled."""

    if not use_cache:
        return None

    if cache_format not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported cache format: {cache_format!r}.")

    paths = get_default_data_paths(data_dir)
    paths.processed.mkdir(parents=True, exist_ok=True)

    stem = cache_name or _default_cache_stem(symbols=symbols, start=start, end=end)
    suffix = ".csv" if cache_format == "csv" else ".parquet"
    return paths.processed / f"{stem}{suffix}"


def _default_cache_stem(
    *,
    symbols: Sequence[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> str:
    """Create a deterministic processed-data cache name."""

    start_text = start.strftime("%Y%m%d") if start is not None else "start"
    end_text = end.strftime("%Y%m%d") if end is not None else "end"
    symbol_text = "_".join(symbol.lower() for symbol in symbols)
    return f"daily_{symbol_text}_{start_text}_{end_text}"


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract a single-symbol frame from a yfinance response."""

    if isinstance(raw.columns, pd.MultiIndex):
        level_names = [name.lower() if name is not None else "" for name in raw.columns.names]

        if "ticker" in level_names:
            ticker_level = level_names.index("ticker")
            frame = raw.xs(symbol, axis=1, level=ticker_level)
        elif "ticker" in {str(value).lower() for value in raw.columns.get_level_values(1)}:
            frame = raw.xs(symbol, axis=1, level=1)
        elif symbol in raw.columns.get_level_values(0):
            frame = raw.xs(symbol, axis=1, level=0)
        elif symbol in raw.columns.get_level_values(-1):
            frame = raw.xs(symbol, axis=1, level=-1)
        else:
            available = sorted({str(value) for value in raw.columns.get_level_values(-1)})
            joined = ", ".join(available)
            raise ValueError(f"Symbol {symbol!r} not found in downloaded data. Available: {joined}.")
        return frame.copy()

    return raw.copy()


def _clean_symbol_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize a single symbol's daily OHLCV data."""

    if frame.empty:
        raise ValueError(f"No rows returned for symbol {symbol}.")

    renamed = frame.rename(columns=YFINANCE_COLUMN_MAP)
    missing_columns = [column for column in YFINANCE_COLUMN_MAP.values() if column not in renamed.columns]

    if "adjusted_close" in missing_columns and "close" in renamed.columns:
        renamed["adjusted_close"] = renamed["close"]
        missing_columns.remove("adjusted_close")

    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Symbol {symbol} is missing required columns: {joined}.")

    cleaned = renamed.loc[:, STANDARD_COLUMNS].copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned.index = cleaned.index.tz_localize(None) if cleaned.index.tz is not None else cleaned.index
    cleaned.index = cleaned.index.normalize()
    cleaned.index.name = "date"
    cleaned = cleaned.sort_index()

    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")

    missing_mask = cleaned[list(STANDARD_COLUMNS[:-1])].isna().any(axis=1) | cleaned["volume"].isna()
    if missing_mask.any():
        cleaned = cleaned.loc[~missing_mask].copy()

    if cleaned.empty:
        raise ValueError(
            f"All rows for symbol {symbol} were removed during validation because of "
            "missing or non-numeric values."
        )

    if (cleaned[["open", "high", "low", "close", "adjusted_close"]] <= 0).any().any():
        raise ValueError(f"Symbol {symbol} contains non-positive price values after cleaning.")

    if (cleaned["volume"] < 0).any():
        raise ValueError(f"Symbol {symbol} contains negative volume after cleaning.")

    cleaned["volume"] = cleaned["volume"].astype("int64")
    cleaned["symbol"] = symbol
    return cleaned


def _save_cached_frames(
    frames: dict[str, pd.DataFrame],
    cache_path: Path,
    cache_format: CacheFormat,
) -> None:
    """Persist cleaned frames to disk."""

    stacked = pd.concat(frames, names=["symbol_key"]).reset_index(level=0, drop=True)

    if cache_format == "csv":
        stacked.to_csv(cache_path)
        return

    try:
        stacked.to_parquet(cache_path)
    except ImportError as exc:
        raise ImportError(
            "Parquet caching requires an installed parquet engine such as pyarrow "
            "or fastparquet."
        ) from exc


def _load_cached_frames(cache_path: Path, cache_format: CacheFormat) -> dict[str, pd.DataFrame]:
    """Load cached frames and restore the per-symbol mapping."""

    if cache_format == "csv":
        cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
    else:
        try:
            cached = pd.read_parquet(cache_path)
        except ImportError as exc:
            raise ImportError(
                "Parquet caching requires an installed parquet engine such as pyarrow "
                "or fastparquet."
            ) from exc
        cached["date"] = pd.to_datetime(cached["date"])
        cached = cached.set_index("date")

    if "symbol" not in cached.columns:
        raise ValueError(f"Cached data at {cache_path} does not contain a symbol column.")

    frames = {
        symbol: frame.loc[:, list(STANDARD_COLUMNS) + ["symbol"]].sort_index()
        for symbol, frame in cached.groupby("symbol", sort=False)
    }

    if not frames:
        raise ValueError(f"Cached data at {cache_path} does not contain any symbol frames.")

    return frames


def _coerce_price_input(
    prices: PriceLike,
    *,
    price_column: str | None,
) -> PriceLike:
    """Accept a price series directly or select a price column from a frame."""

    if isinstance(prices, pd.Series):
        return prices.astype(float)

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Expected a pandas Series or DataFrame for return calculation.")

    if price_column is not None:
        if price_column not in prices.columns:
            raise ValueError(f"Price column {price_column!r} not found in DataFrame.")
        return prices[price_column].astype(float)

    numeric = prices.astype(float)
    return numeric
