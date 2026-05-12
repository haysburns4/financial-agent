from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncConnection

from src.config import settings
from src.db import get_connection, health_check
from src.etrade.auth import auth
from src.models import Indicator, PipelineRun, PriceBar, Signal
from src.scheduler import price_pipeline


class CompleteAuthBody(BaseModel):
    verifier: str


def create_app() -> FastAPI:
    app = FastAPI(title="financial-agent", version="0.1.0")

    @app.get("/health")
    async def health():
        ok = await health_check()
        return {"status": "ok" if ok else "degraded", "db": ok}

    @app.post("/auth/start")
    async def auth_start():
        url = auth.start_auth()
        return {"auth_url": url}

    @app.post("/auth/complete")
    async def auth_complete(body: CompleteAuthBody):
        auth.complete_auth(body.verifier)
        return {"authenticated": True}

    @app.get("/auth/status")
    async def auth_status():
        return {
            "authenticated": auth.is_authenticated(),
            "session_age_minutes": auth.session_age_minutes(),
        }

    @app.post("/pipeline/price/run")
    async def trigger_price_run():
        if not auth.is_authenticated():
            raise HTTPException(status_code=401, detail="E-Trade not authenticated")
        result = await price_pipeline.run(settings.WATCHLIST)
        return {
            "status": result.status,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
            "tickers_processed": result.tickers_processed,
            "tickers_skipped": result.tickers_skipped,
            "bars_stored": result.bars_stored,
            "bars_flagged": result.bars_flagged,
            "errors": result.errors,
        }

    @app.get("/pipeline/runs")
    async def list_pipeline_runs(
        limit: int = 20,
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        stmt = (
            select(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .limit(limit)
        )
        rows = (await conn.execute(stmt)).all()
        return [
            {
                "id": r.id,
                "pipeline": r.pipeline,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "status": r.status,
                "tickers_processed": r.tickers_processed,
                "errors": r.errors,
            }
            for r in rows
        ]

    @app.get("/prices/{ticker}")
    async def latest_prices(
        ticker: str,
        limit: int = 100,
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        stmt = (
            select(PriceBar, Indicator)
            .outerjoin(
                Indicator,
                and_(
                    PriceBar.ticker == Indicator.ticker,
                    PriceBar.timestamp == Indicator.timestamp,
                ),
            )
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
                "data_quality": r.data_quality,
                "indicators": {
                    "rsi_14": r.rsi_14,
                    "macd_line": r.macd_line,
                    "macd_signal": r.macd_signal,
                    "macd_hist": r.macd_hist,
                    "ema_9": r.ema_9,
                    "ema_21": r.ema_21,
                },
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
