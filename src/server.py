from fastapi import Depends, FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from src.db import get_connection, health_check
from src.models import PriceBar, Signal


def create_app() -> FastAPI:
    app = FastAPI(title="financial-agent", version="0.1.0")

    @app.get("/health")
    async def health():
        ok = await health_check()
        return {"status": "ok" if ok else "degraded", "db": ok}

    @app.get("/prices/{ticker}")
    async def latest_prices(
        ticker: str,
        limit: int = 100,
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        stmt = (
            select(PriceBar)
            .where(PriceBar.ticker == ticker.upper())
            .order_by(PriceBar.timestamp.desc())
            .limit(limit)
        )
        rows = (await conn.execute(stmt)).all()
        return [
            {
                "ticker": r.ticker,
                "timestamp": r.timestamp.isoformat(),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "adjusted_close": r.adjusted_close,
            }
            for r in rows
        ]

    @app.get("/signals")
    async def recent_signals(
        limit: int = 50,
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        stmt = select(Signal).order_by(Signal.created_at.desc()).limit(limit)
        rows = (await conn.execute(stmt)).all()
        return [
            {
                "ticker": r.ticker,
                "timestamp": r.timestamp.isoformat(),
                "signal_type": r.signal_type,
                "direction": r.direction,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "delivered": r.delivered,
            }
            for r in rows
        ]

    return app


async def _conn_dep():
    async with get_connection() as conn:
        yield conn
