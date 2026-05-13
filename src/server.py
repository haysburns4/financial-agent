from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncConnection

from src.db import engine, get_connection, health_check
from src.etrade.auth import auth
from src.models import Indicator, PipelineRun, Position, PriceBar, Signal
from src.pipelines import monitored_tickers
from src.scheduler import portfolio_pipeline, price_pipeline
from src.signals.engine import SIGNAL_CATEGORIES, signal_types_for_category


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
        await auth.complete_auth(body.verifier)
        return {"authenticated": True}

    @app.post("/auth/logout")
    async def auth_logout():
        await auth.clear_persisted()
        return {"authenticated": False}

    @app.get("/auth/status")
    async def auth_status():
        return {
            "authenticated": await auth.is_authenticated(),
            "session_age_minutes": auth.session_age_minutes(),
        }

    @app.post("/pipeline/price/run")
    async def trigger_price_run(tickers: str | None = None):
        if not await auth.is_authenticated():
            raise HTTPException(status_code=401, detail="E-Trade not authenticated")
        if tickers:
            ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        else:
            ticker_list = await monitored_tickers(engine)
        result = await price_pipeline.run(ticker_list)
        return {
            "status": result.status,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
            "tickers": ticker_list,
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
        ticker: str | None = None,
        type: str | None = Query(default=None, description="entry | exit | risk"),
        limit: int = 50,
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        stmt = select(Signal).order_by(Signal.created_at.desc())
        if ticker:
            stmt = stmt.where(Signal.ticker == ticker.strip().upper())
        if type:
            allowed = signal_types_for_category(type)
            if not allowed:
                raise HTTPException(
                    status_code=400,
                    detail=f"type must be one of: {sorted(set(SIGNAL_CATEGORIES.values()))}",
                )
            stmt = stmt.where(Signal.signal_type.in_(allowed))
        stmt = stmt.limit(limit)
        rows = (await conn.execute(stmt)).all()
        return [
            {
                "id": r.id,
                "ticker": r.ticker,
                "timestamp": r.timestamp.isoformat(),
                "signal_type": r.signal_type,
                "category": SIGNAL_CATEGORIES.get(r.signal_type),
                "direction": r.direction,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "delivered": r.delivered,
            }
            for r in rows
        ]

    @app.get("/portfolio")
    async def list_positions(
        account_id: str | None = None,
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        stmt = select(Position).order_by(Position.market_value.desc())
        if account_id is not None:
            stmt = stmt.where(Position.account_id == account_id)
        rows = (await conn.execute(stmt)).all()
        positions = [_position_dict(r) for r in rows]

        if account_id is not None:
            return positions

        grouped: dict[str, dict] = {}
        for p in positions:
            bucket = grouped.setdefault(
                p["account_id"],
                {"positions": [], "position_count": 0, "total_market_value": 0.0},
            )
            bucket["positions"].append(p)
            bucket["position_count"] += 1
            bucket["total_market_value"] += p["market_value"]
        return dict(sorted(grouped.items()))

    @app.get("/portfolio/risk")
    async def portfolio_risk(conn: AsyncConnection = Depends(_conn_dep)):
        stmt = select(Position)
        rows = (await conn.execute(stmt)).all()
        if not rows:
            return {"combined": _risk_for([]), "by_account": {}}

        by_account: dict[str, list] = {}
        for r in rows:
            by_account.setdefault(r.account_id, []).append(r)

        return {
            "combined": _risk_for(rows),
            "by_account": {acct: _risk_for(positions) for acct, positions in sorted(by_account.items())},
        }

    @app.post("/pipeline/portfolio/run")
    async def trigger_portfolio_run():
        if not await auth.is_authenticated():
            raise HTTPException(status_code=401, detail="E-Trade not authenticated")
        result = await portfolio_pipeline.run()
        return {
            "status": result.status,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
            "accounts_processed": result.accounts_processed,
            "positions_stored": result.positions_stored,
            "accounts": result.accounts,
            "errors": result.errors,
        }

    return app


async def _conn_dep():
    async with get_connection() as conn:
        yield conn


def _position_dict(r) -> dict:
    pnl = r.market_value - r.cost_basis
    return {
        "account_id": r.account_id,
        "ticker": r.ticker,
        "quantity": r.quantity,
        "cost_basis": r.cost_basis,
        "market_value": r.market_value,
        "pnl": pnl,
        "pnl_pct": (pnl / r.cost_basis) if r.cost_basis else None,
        "last_updated": r.last_updated.isoformat(),
    }


def _risk_for(positions: list) -> dict:
    """Risk metrics for an arbitrary set of positions (one account or combined)."""
    total_exposure = sum(p.market_value for p in positions)
    total_cost = sum(p.cost_basis for p in positions)
    concentration = sorted(
        (
            {
                "ticker": p.ticker,
                "market_value": p.market_value,
                "pct_of_portfolio": (p.market_value / total_exposure) if total_exposure else 0.0,
            }
            for p in positions
        ),
        key=lambda x: x["pct_of_portfolio"],
        reverse=True,
    )
    # Proxy for true peak-to-trough drawdown until we persist portfolio snapshots.
    drawdown = (total_cost - total_exposure) / total_cost if total_cost > 0 else None
    return {
        "total_exposure": total_exposure,
        "concentration": concentration,
        "drawdown": drawdown,
        "position_count": len(positions),
    }
