"""Price data pipeline: fetch quotes, validate, upsert bars + indicators."""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.etrade.market import ETradeMarketClient
from src.models import Indicator, PipelineRun, PriceBar
from src.pipelines.validators import RawPriceBar, ValidatedPriceBar
from src.signals.technical import compute_indicators


@dataclass
class PipelineRunResult:
    started_at: datetime
    completed_at: datetime
    status: str
    tickers_processed: int
    tickers_skipped: int
    bars_stored: int
    bars_flagged: int
    errors: list[str] = field(default_factory=list)


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    skip_runs_remaining: int = 0


class PricePipeline:
    FAILURE_THRESHOLD = 3
    SKIP_RUNS = 5
    HISTORY_LIMIT = 50

    def __init__(self, market: ETradeMarketClient, engine: AsyncEngine) -> None:
        self._market = market
        self._engine = engine
        self._circuit: dict[str, _CircuitState] = {}

    async def run(self, tickers: list[str]) -> PipelineRunResult:
        started = datetime.now(timezone.utc)
        processed = 0
        skipped = 0
        bars_stored = 0
        bars_flagged = 0
        errors: list[str] = []

        logger.info("price_pipeline starting for {} tickers", len(tickers))

        async with self._engine.begin() as conn:
            run_id = (
                await conn.execute(
                    sqlite_insert(PipelineRun).values(
                        pipeline="price_pipeline",
                        started_at=started,
                        status="running",
                    )
                )
            ).inserted_primary_key[0]

            for ticker in tickers:
                if self._is_circuit_open(ticker):
                    self._tick_circuit(ticker)
                    logger.info(
                        "Skipping {} (circuit open, {} runs remaining)",
                        ticker,
                        self._circuit[ticker].skip_runs_remaining,
                    )
                    skipped += 1
                    continue

                try:
                    async with conn.begin_nested():
                        stored, flagged = await self._process_ticker(conn, ticker)
                        bars_stored += stored
                        bars_flagged += flagged
                        processed += 1
                        self._reset_circuit(ticker)
                except Exception as exc:
                    logger.exception("price_pipeline failed for {}", ticker)
                    errors.append(f"{ticker}: {exc}")
                    self._record_failure(ticker)

            completed = datetime.now(timezone.utc)
            if errors and processed == 0:
                status = "failed"
            elif errors:
                status = "partial"
            else:
                status = "ok"

            await conn.execute(
                update(PipelineRun)
                .where(PipelineRun.id == run_id)
                .values(
                    completed_at=completed,
                    status=status,
                    tickers_processed=processed,
                    errors="\n".join(errors) if errors else None,
                )
            )

        result = PipelineRunResult(
            started_at=started,
            completed_at=completed,
            status=status,
            tickers_processed=processed,
            tickers_skipped=skipped,
            bars_stored=bars_stored,
            bars_flagged=bars_flagged,
            errors=errors,
        )
        logger.info(
            "price_pipeline done: status={} processed={} skipped={} stored={} flagged={} errors={}",
            result.status,
            result.tickers_processed,
            result.tickers_skipped,
            result.bars_stored,
            result.bars_flagged,
            len(result.errors),
        )
        return result

    async def _process_ticker(
        self, conn: AsyncConnection, ticker: str
    ) -> tuple[int, int]:
        logger.info("Fetching {}...", ticker)
        quotes = await self._market.get_quotes([ticker])
        if not quotes:
            raise RuntimeError(f"no quotes returned for {ticker}")

        raw_bars: list[RawPriceBar] = []
        for q in quotes:
            try:
                # E-Trade snapshots are not finalized OHLC bars: lastTrade can sit
                # outside [low, high] when a new extreme is being printed, and the
                # sandbox returns zero for fields it doesn't model (open/low/high).
                # Treat non-positive fields as missing, then enforce mutual
                # consistency so the strict OHLC validator doesn't reject the bar.
                last = q["last"] if q["last"] > 0 else 0.0
                prev_close = q["close"] if q["close"] > 0 else 0.0
                close = last or prev_close
                if close <= 0:
                    logger.warning("E-Trade returned no usable price for {}; skipping", ticker)
                    continue
                open_ = q["open"] if q["open"] > 0 else close
                high = max(q["high"], open_, close)
                low_candidates = [v for v in (q["low"], open_, close) if v > 0]
                low = min(low_candidates)
                raw_bars.append(
                    RawPriceBar(
                        ticker=q["ticker"],
                        timestamp=q["timestamp"],
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=q["volume"],
                    )
                )
            except Exception as exc:
                logger.warning("validation rejected raw bar for {}: {}", ticker, exc)

        if not raw_bars:
            raise RuntimeError(f"no valid bars after validation for {ticker}")

        prev_close = await self._previous_close(conn, ticker)
        validated = [
            ValidatedPriceBar.from_raw(r, prev_close=prev_close) for r in raw_bars
        ]
        flagged_count = sum(1 for v in validated if v.anomalies)
        logger.info(
            "Validated {} bar(s) for {}, {} anomalies",
            len(validated),
            ticker,
            flagged_count,
        )

        await self._upsert_bars(conn, validated)
        logger.info("Upserted {} bar(s) to DB for {}", len(validated), ticker)

        history = await self._load_history(conn, ticker, self.HISTORY_LIMIT)
        if history.empty:
            logger.info("No history for {}; skipping indicators", ticker)
        else:
            indicators_df = compute_indicators(history)
            await self._upsert_indicators(conn, ticker, indicators_df)
            logger.info("Upserted {} indicator row(s) for {}", len(indicators_df), ticker)

        return len(validated), flagged_count

    async def _previous_close(
        self, conn: AsyncConnection, ticker: str
    ) -> float | None:
        stmt = (
            select(PriceBar.close)
            .where(PriceBar.ticker == ticker)
            .order_by(PriceBar.timestamp.desc())
            .limit(1)
        )
        row = (await conn.execute(stmt)).first()
        return float(row[0]) if row else None

    async def _load_history(
        self, conn: AsyncConnection, ticker: str, limit: int
    ) -> pd.DataFrame:
        stmt = (
            select(PriceBar)
            .where(PriceBar.ticker == ticker)
            .order_by(PriceBar.timestamp.desc())
            .limit(limit)
        )
        rows = (await conn.execute(stmt)).all()
        if not rows:
            return pd.DataFrame()
        records = [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in reversed(rows)  # chronological for indicators
        ]
        return pd.DataFrame(records)

    async def _upsert_bars(
        self, conn: AsyncConnection, validated: list[ValidatedPriceBar]
    ) -> None:
        rows = [
            {
                "ticker": v.ticker,
                "timestamp": v.timestamp,
                "open": v.open,
                "high": v.high,
                "low": v.low,
                "close": v.close,
                "volume": v.volume,
                "adjusted_close": v.adjusted_close,
                "data_quality": v.data_quality,
            }
            for v in validated
        ]
        stmt = sqlite_insert(PriceBar).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "timestamp"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "adjusted_close": stmt.excluded.adjusted_close,
                "data_quality": stmt.excluded.data_quality,
            },
        )
        await conn.execute(stmt)

    async def _upsert_indicators(
        self, conn: AsyncConnection, ticker: str, df: pd.DataFrame
    ) -> None:
        rows: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": r["timestamp"],
                    "rsi_14": _none_if_nan(r.get("rsi_14")),
                    "macd_line": _none_if_nan(r.get("macd_line")),
                    "macd_signal": _none_if_nan(r.get("macd_signal")),
                    "macd_hist": _none_if_nan(r.get("macd_hist")),
                    "ema_9": _none_if_nan(r.get("ema_9")),
                    "ema_21": _none_if_nan(r.get("ema_21")),
                }
            )
        if not rows:
            return
        stmt = sqlite_insert(Indicator).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "timestamp"],
            set_={
                "rsi_14": stmt.excluded.rsi_14,
                "macd_line": stmt.excluded.macd_line,
                "macd_signal": stmt.excluded.macd_signal,
                "macd_hist": stmt.excluded.macd_hist,
                "ema_9": stmt.excluded.ema_9,
                "ema_21": stmt.excluded.ema_21,
            },
        )
        await conn.execute(stmt)

    def _is_circuit_open(self, ticker: str) -> bool:
        state = self._circuit.get(ticker)
        return state is not None and state.skip_runs_remaining > 0

    def _tick_circuit(self, ticker: str) -> None:
        state = self._circuit[ticker]
        state.skip_runs_remaining -= 1
        if state.skip_runs_remaining <= 0:
            state.consecutive_failures = 0  # give it another shot next run

    def _record_failure(self, ticker: str) -> None:
        state = self._circuit.setdefault(ticker, _CircuitState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.FAILURE_THRESHOLD:
            state.skip_runs_remaining = self.SKIP_RUNS
            logger.warning(
                "Circuit opened for {} after {} failures; skipping next {} runs",
                ticker,
                state.consecutive_failures,
                self.SKIP_RUNS,
            )

    def _reset_circuit(self, ticker: str) -> None:
        if ticker in self._circuit:
            self._circuit[ticker] = _CircuitState()


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f
