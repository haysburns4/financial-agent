"""Historical backfill pipeline backed by yfinance.

This is a one-time / on-demand operation, not on the scheduler. For each
ticker it pulls two slices from yfinance:
  - daily bars over `period_daily` (default 2y) at 1d intervals
  - intraday bars over `period_intraday` (default 60d, yfinance max) at 5m

Both are validated through the same `RawPriceBar` validator the live pipeline
uses, then bulk-upserted into `price_bars` keyed on (ticker, timestamp). After
both slices land, indicators are recomputed across the entire stored history
for the ticker so warmup artifacts from early bars get overwritten with stable
values.
"""
import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.models import Indicator, PriceBar
from src.pipelines.validators import RawPriceBar
from src.signals.technical import compute_indicators


_BAR_CHUNK = 500            # rows per upsert statement (SQLite variable limit)
_RATE_LIMIT_SECONDS = 0.5   # sleep between tickers — be kind to yfinance


@dataclass
class BackfillResult:
    tickers_processed: int = 0
    daily_bars_added: int = 0
    intraday_bars_added: int = 0
    indicators_computed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _flatten_yf_frame(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns even for single-ticker calls in
    1.x; flatten to single-level Open/High/Low/Close/Volume."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _yf_to_rows(ticker: str, df: pd.DataFrame, *, is_intraday: bool) -> list[dict]:
    """Validate one yfinance frame through RawPriceBar and return upsert rows.

    Bars that fail validation are logged and dropped; we never abort the slice.
    Daily timestamps come in tz-naive; we localize to UTC. Intraday timestamps
    come tz-aware (typically America/New_York) and we convert to UTC.
    """
    if df.empty:
        return []
    df = _flatten_yf_frame(df).reset_index()
    df = df.rename(columns={
        df.columns[0]: "timestamp",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    ts_col = df["timestamp"]
    if getattr(ts_col.dt, "tz", None) is None:
        df["timestamp"] = ts_col.dt.tz_localize("UTC")
    else:
        df["timestamp"] = ts_col.dt.tz_convert("UTC")

    rows: list[dict] = []
    rejected = 0
    for _, r in df.iterrows():
        try:
            bar = RawPriceBar(
                ticker=ticker,
                timestamp=r["timestamp"].to_pydatetime(),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
                # auto_adjust=True means Close is already adjusted; carry it through.
                adjusted_close=float(r["close"]),
            )
        except (ValidationError, ValueError, TypeError) as exc:
            rejected += 1
            logger.debug("backfill rejected bar {} {}: {}", ticker, r["timestamp"], exc)
            continue
        rows.append({
            "ticker": bar.ticker,
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "adjusted_close": bar.adjusted_close,
            "data_quality": "ok",
        })
    if rejected:
        logger.warning(
            "{}: dropped {} bar(s) failing validation ({})",
            ticker,
            rejected,
            "intraday" if is_intraday else "daily",
        )
    return rows


async def _chunked_upsert(
    conn: AsyncConnection,
    model,
    rows: list[dict],
    conflict_cols: list[str],
    update_cols: list[str],
    chunk: int = _BAR_CHUNK,
) -> int:
    if not rows:
        return 0
    written = 0
    for i in range(0, len(rows), chunk):
        batch = rows[i:i + chunk]
        stmt = sqlite_insert(model).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=conflict_cols,
            set_={col: getattr(stmt.excluded, col) for col in update_cols},
        )
        await conn.execute(stmt)
        written += len(batch)
    return written


class BackfillPipeline:
    DAILY_PERIOD = "2y"
    INTRADAY_PERIOD = "60d"
    INTRADAY_INTERVAL = "5m"

    PRICE_UPDATE_COLS = [
        "open", "high", "low", "close", "volume", "adjusted_close", "data_quality",
    ]
    INDICATOR_UPDATE_COLS = [
        "rsi_14", "macd_line", "macd_signal", "macd_hist", "ema_9", "ema_21",
    ]

    def __init__(self, engine: AsyncEngine, settings=None) -> None:
        self._engine = engine
        self._settings = settings  # kept for symmetry with other pipelines; unused

    async def run(
        self,
        tickers: list[str],
        period_daily: str = DAILY_PERIOD,
        period_intraday: str = INTRADAY_PERIOD,
        interval_intraday: str = INTRADAY_INTERVAL,
    ) -> BackfillResult:
        started = time.monotonic()
        result = BackfillResult()
        if not tickers:
            return result

        for i, ticker in enumerate(tickers):
            if i > 0:
                await asyncio.sleep(_RATE_LIMIT_SECONDS)
            ticker = ticker.strip().upper()
            try:
                await self._backfill_one(
                    ticker, period_daily, period_intraday, interval_intraday, result,
                )
                result.tickers_processed += 1
            except Exception as exc:
                logger.exception("backfill failed for {}", ticker)
                result.errors.append(f"{ticker}: {exc}")

        result.duration_seconds = time.monotonic() - started
        logger.info(
            "Backfill complete: {} tickers, {} daily bars, {} intraday bars, "
            "{} indicators, {} errors, {:.1f}s",
            result.tickers_processed,
            result.daily_bars_added,
            result.intraday_bars_added,
            result.indicators_computed,
            len(result.errors),
            result.duration_seconds,
        )
        return result

    async def _backfill_one(
        self,
        ticker: str,
        period_daily: str,
        period_intraday: str,
        interval_intraday: str,
        result: BackfillResult,
    ) -> None:
        # --- Daily ---
        logger.info("Backfilling {}: fetching {} daily bars...", ticker, period_daily)
        daily_df = await asyncio.to_thread(
            yf.download,
            ticker,
            period=period_daily,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        daily_rows = _yf_to_rows(ticker, daily_df, is_intraday=False)

        async with self._engine.begin() as conn:
            daily_written = await _chunked_upsert(
                conn, PriceBar, daily_rows,
                conflict_cols=["ticker", "timestamp"],
                update_cols=self.PRICE_UPDATE_COLS,
            )
        result.daily_bars_added += daily_written
        logger.info(
            "{} daily: {} bars fetched, {} upserted",
            ticker, len(daily_rows), daily_written,
        )

        # --- Intraday ---
        logger.info(
            "Backfilling {}: fetching {} intraday bars at {}...",
            ticker, period_intraday, interval_intraday,
        )
        intraday_df = await asyncio.to_thread(
            yf.download,
            ticker,
            period=period_intraday,
            interval=interval_intraday,
            auto_adjust=True,
            progress=False,
        )
        intraday_rows = _yf_to_rows(ticker, intraday_df, is_intraday=True)
        async with self._engine.begin() as conn:
            intraday_written = await _chunked_upsert(
                conn, PriceBar, intraday_rows,
                conflict_cols=["ticker", "timestamp"],
                update_cols=self.PRICE_UPDATE_COLS,
            )
        result.intraday_bars_added += intraday_written
        logger.info(
            "{} intraday: {} bars fetched, {} upserted",
            ticker, len(intraday_rows), intraday_written,
        )

        # --- Indicators across full history ---
        indicators_written = await self._recompute_indicators(ticker)
        result.indicators_computed += indicators_written
        logger.info("{} indicators: computed on {} total bars", ticker, indicators_written)

    async def _recompute_indicators(self, ticker: str) -> int:
        # Load full history (ordered ASC), compute indicators, upsert all rows.
        async with self._engine.connect() as conn:
            stmt = (
                select(
                    PriceBar.timestamp,
                    PriceBar.open,
                    PriceBar.high,
                    PriceBar.low,
                    PriceBar.close,
                    PriceBar.volume,
                )
                .where(PriceBar.ticker == ticker)
                .order_by(PriceBar.timestamp.asc())
            )
            rows = (await conn.execute(stmt)).all()
        if not rows:
            return 0

        df = pd.DataFrame(
            {
                "timestamp": [r.timestamp for r in rows],
                "open": [r.open for r in rows],
                "high": [r.high for r in rows],
                "low": [r.low for r in rows],
                "close": [r.close for r in rows],
                "volume": [r.volume for r in rows],
            }
        )
        df = await asyncio.to_thread(compute_indicators, df)

        indicator_rows: list[dict] = []
        for _, r in df.iterrows():
            ts = r["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            indicator_rows.append({
                "ticker": ticker,
                "timestamp": ts,
                "rsi_14": _none_if_nan(r.get("rsi_14")),
                "macd_line": _none_if_nan(r.get("macd_line")),
                "macd_signal": _none_if_nan(r.get("macd_signal")),
                "macd_hist": _none_if_nan(r.get("macd_hist")),
                "ema_9": _none_if_nan(r.get("ema_9")),
                "ema_21": _none_if_nan(r.get("ema_21")),
            })

        async with self._engine.begin() as conn:
            return await _chunked_upsert(
                conn, Indicator, indicator_rows,
                conflict_cols=["ticker", "timestamp"],
                update_cols=self.INDICATOR_UPDATE_COLS,
            )
