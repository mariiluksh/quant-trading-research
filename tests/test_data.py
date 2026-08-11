"""Tests for the historical market-data layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research.data import download_daily_ohlcv, log_returns, simple_returns


def test_download_daily_ohlcv_single_symbol_standardizes_and_cleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-symbol downloads should be standardized and cleaned."""

    index = pd.to_datetime(
        [
            "2024-01-03 00:00:00+00:00",
            "2024-01-02 00:00:00+00:00",
            "2024-01-02 00:00:00+00:00",
            "2024-01-04 00:00:00+00:00",
        ]
    )
    raw = pd.DataFrame(
        {
            "Open": [101.0, 99.0, 100.0, np.nan],
            "High": [102.0, 100.0, 101.0, 103.0],
            "Low": [100.0, 98.0, 99.0, 101.0],
            "Close": [101.5, 99.5, 100.5, 102.0],
            "Adj Close": [101.4, 99.4, 100.4, 101.9],
            "Volume": [1000, 900, 950, 1100],
        },
        index=index,
    )

    def fake_download(**_: object) -> pd.DataFrame:
        return raw

    monkeypatch.setattr("quant_research.data.yf.download", fake_download)

    result = download_daily_ohlcv("aapl")
    frame = result["AAPL"]

    assert list(frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "symbol",
    ]
    assert frame.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert frame.loc[pd.Timestamp("2024-01-02"), "open"] == 100.0
    assert frame.loc[pd.Timestamp("2024-01-02"), "adjusted_close"] == 100.4
    assert frame["symbol"].unique().tolist() == ["AAPL"]


def test_download_daily_ohlcv_multiple_symbols_preserves_calendars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple symbols should be split into independent clean frames."""

    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Adj Close", "AAPL"),
            ("Volume", "AAPL"),
            ("Open", "MSFT"),
            ("High", "MSFT"),
            ("Low", "MSFT"),
            ("Close", "MSFT"),
            ("Adj Close", "MSFT"),
            ("Volume", "MSFT"),
        ],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [
            [100, 101, 99, 100.5, 100.4, 1000, 200, 201, 199, 200.5, 200.4, 2000],
            [101, 102, 100, 101.5, 101.4, 1100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [102, 103, 101, 102.5, 102.4, 1200, 202, 203, 201, 202.5, 202.4, 2200],
        ],
        index=index,
        columns=columns,
    )

    def fake_download(**_: object) -> pd.DataFrame:
        return raw

    monkeypatch.setattr("quant_research.data.yf.download", fake_download)

    result = download_daily_ohlcv(["AAPL", "MSFT"])

    assert set(result) == {"AAPL", "MSFT"}
    assert result["AAPL"].index.tolist() == index.tolist()
    assert result["MSFT"].index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-04"),
    ]


def test_download_daily_ohlcv_uses_csv_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cached processed data should be reused instead of downloading again."""

    raw = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.2, 11.2],
            "Adj Close": [10.1, 11.1],
            "Volume": [100, 110],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    calls = {"count": 0}

    def fake_download(**_: object) -> pd.DataFrame:
        calls["count"] += 1
        return raw

    monkeypatch.setattr("quant_research.data.yf.download", fake_download)

    first = download_daily_ohlcv(
        "AAPL",
        use_cache=True,
        cache_format="csv",
        cache_name="test_cache",
        data_dir=tmp_path,
    )

    def fail_download(**_: object) -> pd.DataFrame:
        raise AssertionError("Download should not be called when cache exists.")

    monkeypatch.setattr("quant_research.data.yf.download", fail_download)

    second = download_daily_ohlcv(
        "AAPL",
        use_cache=True,
        cache_format="csv",
        cache_name="test_cache",
        data_dir=tmp_path,
    )

    assert calls["count"] == 1
    pd.testing.assert_frame_equal(first["AAPL"], second["AAPL"])


def test_download_daily_ohlcv_rejects_invalid_bounds() -> None:
    """Start date must be earlier than end date."""

    with pytest.raises(ValueError, match="Expected start date before end date"):
        download_daily_ohlcv("AAPL", start="2024-01-05", end="2024-01-05")


def test_simple_and_log_returns_use_explicit_missing_behavior() -> None:
    """Return helpers should not fill gaps before calculating returns."""

    prices = pd.Series(
        [100.0, np.nan, 121.0, 133.1],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        name="adjusted_close",
    )

    simple = simple_returns(prices)
    log_ret = log_returns(prices)

    assert np.isnan(simple.iloc[0])
    assert np.isnan(simple.iloc[1])
    assert np.isnan(simple.iloc[2])
    assert simple.iloc[3] == pytest.approx(0.1)

    assert np.isnan(log_ret.iloc[0])
    assert np.isnan(log_ret.iloc[1])
    assert np.isnan(log_ret.iloc[2])
    assert log_ret.iloc[3] == pytest.approx(np.log(1.1))


def test_simple_returns_can_select_adjusted_close_column() -> None:
    """Return helpers should work directly from cleaned symbol frames."""

    frame = pd.DataFrame(
        {
            "close": [100.0, 90.0],
            "adjusted_close": [100.0, 99.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    returns = simple_returns(frame, price_column="adjusted_close")
    assert returns.iloc[1] == pytest.approx(-0.01)
