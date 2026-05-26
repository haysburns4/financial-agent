"""Discord webhook alerts.

Two entry points:

- `deliver_pending_signals(synthesizer=...)` — runs in the per-tick scheduler
  chain. Posts raw per-signal alerts as they fire so the user gets real-time
  notifications. If a synthesizer is passed, attempts a single synthesized
  briefing for the batch and falls back to raw per-signal on any failure.
- `post_daily_briefing(synthesizer)` — runs once a day, just after market
  close. Pulls the last 24h of signals (regardless of delivered state) and
  posts a standalone synthesized digest. Does NOT touch the `delivered` flag
  — those signals were already posted raw as they fired.
"""
from datetime import datetime, timedelta, timezone

import httpx
from loguru import logger
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from src.config import settings
from src.db import get_connection
from src.models import Indicator, Position, PriceBar, Signal


_DISCORD_LIMIT = 1900  # Discord caps at 2000; leave headroom for prefix
_BARS_PER_TICKER = 5


def _format(row) -> str:
    direction = f" {row.direction}" if row.direction else ""
    return (
        f"**{row.signal_type.upper()}{direction}** — `{row.ticker}` "
        f"(confidence {row.confidence:.0%})\n"
        f"{row.reasoning}"
    )


async def post(content: str) -> bool:
    if not settings.DISCORD_WEBHOOK_URL:
        logger.debug("Discord webhook not configured — skipping alert")
        return False
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                settings.DISCORD_WEBHOOK_URL,
                json={"content": content[:_DISCORD_LIMIT]},
            )
            resp.raise_for_status()
        return True
    except Exception:
        logger.exception("Discord webhook post failed")
        return False


async def deliver_pending_signals(synthesizer=None) -> int:
    """Deliver undelivered signals via Discord. Returns the count marked delivered.

    If a `synthesizer` is passed, attempt a single synthesized post covering
    the whole batch; on any failure (synthesis or webhook post) fall back to
    the original per-signal flow. Without a synthesizer, behaves identically
    to the pre-existing per-signal flow.
    """
    async with get_connection() as conn:
        stmt = (
            select(Signal)
            .where(Signal.delivered.is_(False))
            .order_by(Signal.created_at)
        )
        rows = (await conn.execute(stmt)).all()
        if not rows:
            return 0

        if synthesizer is not None:
            if await _try_synthesized_post(conn, rows, synthesizer):
                return len(rows)
            logger.warning("synthesis path failed; falling back to per-signal delivery")

        return await _deliver_per_signal(conn, rows)


async def _deliver_per_signal(conn: AsyncConnection, rows) -> int:
    sent = 0
    for row in rows:
        if await post(_format(row)):
            await conn.execute(
                update(Signal).where(Signal.id == row.id).values(delivered=True)
            )
            sent += 1
    return sent


async def _try_synthesized_post(conn: AsyncConnection, rows, synthesizer) -> bool:
    signal_dicts = [
        {
            "id": r.id,
            "ticker": r.ticker,
            "timestamp": r.timestamp,
            "signal_type": r.signal_type,
            "direction": r.direction,
            "confidence": r.confidence,
            "reasoning": r.reasoning,
        }
        for r in rows
    ]
    try:
        context = await build_synthesis_context(conn, signal_dicts)
        narrative = await synthesizer.synthesize(signal_dicts, context)
    except Exception:
        logger.exception("synthesizer raised; will fall back to raw delivery")
        return False
    if not narrative:
        return False

    header = f"**{len(rows)} new signal(s) — synthesized briefing:**\n\n"
    if not await post(header + narrative):
        return False

    for row in rows:
        await conn.execute(
            update(Signal).where(Signal.id == row.id).values(delivered=True)
        )
    return True


async def post_daily_briefing(synthesizer) -> bool:
    """Synthesize the last 24h of signals and post one standalone Discord briefing.

    Independent of per-tick raw delivery: this does NOT mutate the `delivered`
    flag because those signals were already posted individually as they fired.
    Skips silently if no signals fired in the last 24h.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    async with get_connection() as conn:
        rows = (
            await conn.execute(
                select(Signal)
                .where(Signal.created_at >= cutoff)
                .order_by(Signal.created_at)
            )
        ).all()
        if not rows:
            logger.info("daily briefing: no signals in last 24h; skipping")
            return False

        signal_dicts = [
            {
                "id": r.id,
                "ticker": r.ticker,
                "timestamp": r.timestamp,
                "signal_type": r.signal_type,
                "direction": r.direction,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
            }
            for r in rows
        ]
        try:
            context = await build_synthesis_context(conn, signal_dicts)
            narrative = await synthesizer.synthesize(signal_dicts, context)
        except Exception:
            logger.exception("daily briefing synthesis failed")
            return False
        if not narrative:
            return False

    header = f"**Daily briefing — {len(rows)} signal(s) in the last 24h:**\n\n"
    posted = await post(header + narrative)
    if posted:
        logger.info("daily briefing posted ({} signal(s) summarized)", len(rows))
    return posted


async def build_synthesis_context(
    conn: AsyncConnection, signals: list[dict]
) -> dict:
    """Gather positions + last-5 bars per affected ticker."""
    tickers = sorted({s["ticker"] for s in signals if s.get("ticker")})

    positions_stmt = (
        select(Position)
        .where(Position.ticker.in_(tickers))
        .order_by(Position.ticker)
    )
    positions = [
        {
            "account_id": r.account_id,
            "ticker": r.ticker,
            "quantity": r.quantity,
            "cost_basis": r.cost_basis,
            "market_value": r.market_value,
        }
        for r in (await conn.execute(positions_stmt)).all()
    ]

    bars_by_ticker: dict[str, list[dict]] = {}
    for ticker in tickers:
        stmt = (
            select(
                PriceBar.timestamp, PriceBar.open, PriceBar.high, PriceBar.low,
                PriceBar.close, PriceBar.volume,
                Indicator.rsi_14, Indicator.macd_line, Indicator.macd_signal,
                Indicator.macd_hist, Indicator.ema_9, Indicator.ema_21,
            )
            .outerjoin(
                Indicator,
                and_(
                    PriceBar.ticker == Indicator.ticker,
                    PriceBar.timestamp == Indicator.timestamp,
                ),
            )
            .where(PriceBar.ticker == ticker)
            .order_by(PriceBar.timestamp.desc())
            .limit(_BARS_PER_TICKER)
        )
        rows = (await conn.execute(stmt)).all()
        if not rows:
            continue
        bars_by_ticker[ticker] = [
            {
                "timestamp": r.timestamp,
                "open": r.open, "high": r.high, "low": r.low,
                "close": r.close, "volume": r.volume,
                "rsi_14": r.rsi_14, "macd_line": r.macd_line,
                "macd_signal": r.macd_signal, "macd_hist": r.macd_hist,
                "ema_9": r.ema_9, "ema_21": r.ema_21,
            }
            for r in reversed(rows)
        ]
    return {"positions": positions, "bars_by_ticker": bars_by_ticker}
