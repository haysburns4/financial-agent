from dataclasses import asdict
from datetime import date, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from src.agui import create_agui_router
from src.alerts.discord import build_synthesis_context
from src.backtest import persistence as backtest_persistence
from src.backtest.runner import BacktestRunner
from src.config import settings
from src.db import engine, get_connection, health_check
from src.etrade.auth import ETradeAuthError, auth
from src.models import Indicator, PipelineRun, Position, PriceBar, Signal
from src.pipelines import monitored_tickers
from src.portfolio import dashboard, position_dict, risk_by_account, risk_summary
from src.pipelines.backfill_pipeline import BackfillPipeline
from src.scheduler import (
    agent_chat,
    portfolio_pipeline,
    price_pipeline,
    signal_synthesizer,
)
from src.signals.engine import SIGNAL_CATEGORIES, signal_types_for_category


class CompleteAuthBody(BaseModel):
    verifier: str


class BackfillRequest(BaseModel):
    tickers: list[str] | None = None
    period_daily: str = "2y"
    period_intraday: str = "60d"
    interval_intraday: str = "5m"


class BacktestRequest(BaseModel):
    tickers: list[str] | None = None
    start_date: date
    end_date: date
    forward_window_days: int = 5
    outcome_threshold_pct: float = 0.01


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_history: list[ChatTurn] | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="financial-agent", version="0.1.0")
    app.include_router(create_agui_router(agent_chat))

    @app.get("/health")
    async def health():
        ok = await health_check()
        return {"status": "ok" if ok else "degraded", "db": ok}

    @app.post("/auth/start")
    async def auth_start():
        try:
            url = auth.start_auth()
        except ETradeAuthError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"auth_url": url}

    @app.post("/auth/complete")
    async def auth_complete(body: CompleteAuthBody):
        try:
            await auth.complete_auth(body.verifier)
        except ETradeAuthError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
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
        positions = [position_dict(r) for r in rows]

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

    @app.get("/dashboard")
    async def get_dashboard(conn: AsyncConnection = Depends(_conn_dep)):
        """Portfolio panel payload for the web UI: totals, accounts, auth state."""
        rows = (await conn.execute(select(Position))).all()
        data = dashboard(rows)
        last_updated = data["last_updated"]
        if last_updated is not None and last_updated.tzinfo is None:
            # SQLite drops the offset; positions are always written in UTC.
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        return {
            **data,
            "last_updated": last_updated.isoformat() if last_updated else None,
            "authenticated": await auth.is_authenticated(),
        }

    @app.get("/portfolio/risk")
    async def portfolio_risk(conn: AsyncConnection = Depends(_conn_dep)):
        rows = (await conn.execute(select(Position))).all()
        if not rows:
            return {"combined": risk_summary([]), "by_account": {}}
        return risk_by_account(rows)

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

    @app.post("/pipeline/backfill/run")
    async def trigger_backfill(body: BackfillRequest | None = None):
        body = body or BackfillRequest()
        tickers = body.tickers or await monitored_tickers(engine)
        pipeline = BackfillPipeline(engine, settings)
        from loguru import logger
        logger.info("backfill starting for {} ticker(s): {}", len(tickers), tickers)
        result = await pipeline.run(
            tickers=tickers,
            period_daily=body.period_daily,
            period_intraday=body.period_intraday,
            interval_intraday=body.interval_intraday,
        )
        logger.info(
            "backfill done in {:.1f}s: {} tickers, {} daily, {} intraday, {} indicators, {} errors",
            result.duration_seconds, result.tickers_processed,
            result.daily_bars_added, result.intraday_bars_added,
            result.indicators_computed, len(result.errors),
        )
        return asdict(result)

    @app.get("/prices/{ticker}/coverage")
    async def price_coverage(
        ticker: str,
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        ticker = ticker.strip().upper()
        # Total bar count + earliest / latest.
        bar_summary = (
            await conn.execute(
                select(
                    func.count(PriceBar.id),
                    func.min(PriceBar.timestamp),
                    func.max(PriceBar.timestamp),
                )
                .where(PriceBar.ticker == ticker)
            )
        ).first()
        total_bars = bar_summary[0] or 0
        earliest = bar_summary[1]
        latest = bar_summary[2]
        if total_bars == 0:
            return {
                "ticker": ticker,
                "daily_bars": 0,
                "intraday_bars": 0,
                "earliest_bar": None,
                "latest_bar": None,
                "indicator_coverage": 0.0,
            }

        # Crude daily vs intraday split: anything timestamped exactly on a UTC
        # date boundary (00:00:00) is treated as a daily bar; everything else
        # is intraday. Backfill_pipeline stamps daily bars at UTC midnight, so
        # this matches its semantics.
        daily_count = (
            await conn.execute(
                select(func.count(PriceBar.id))
                .where(PriceBar.ticker == ticker)
                .where(func.strftime("%H:%M:%S", PriceBar.timestamp) == "00:00:00")
            )
        ).scalar() or 0
        intraday_count = total_bars - daily_count

        indicator_count = (
            await conn.execute(
                select(func.count(Indicator.id)).where(Indicator.ticker == ticker)
            )
        ).scalar() or 0
        coverage = (indicator_count / total_bars) if total_bars else 0.0

        def _iso(ts):
            if ts is None:
                return None
            if ts.tzinfo is None:
                from datetime import timezone as _tz
                ts = ts.replace(tzinfo=_tz.utc)
            return ts.isoformat()

        return {
            "ticker": ticker,
            "daily_bars": daily_count,
            "intraday_bars": intraday_count,
            "earliest_bar": _iso(earliest),
            "latest_bar": _iso(latest),
            "indicator_coverage": round(coverage, 4),
        }

    @app.post("/backtest/run")
    async def trigger_backtest(body: BacktestRequest):
        tickers = body.tickers or await monitored_tickers(engine)
        runner = BacktestRunner(engine)
        report = await runner.run(
            tickers=tickers,
            start_date=body.start_date,
            end_date=body.end_date,
            forward_window_days=body.forward_window_days,
            outcome_threshold_pct=body.outcome_threshold_pct,
        )
        backtest_persistence.save_report(report)
        return asdict(report)

    @app.get("/backtest/latest")
    async def latest_backtest():
        report = backtest_persistence.load_report_json()
        if report is None:
            raise HTTPException(status_code=404, detail="no backtest report on disk")
        return report

    @app.post("/backtest/apply")
    async def apply_backtest():
        try:
            return backtest_persistence.apply_latest_report()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/chat")
    async def chat(body: ChatRequest):
        history = (
            [{"role": t.role, "content": t.content} for t in body.conversation_history]
            if body.conversation_history
            else None
        )
        try:
            return await agent_chat.ask(body.question, history)
        except Exception as exc:
            from loguru import logger
            logger.exception("/chat failed")
            raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

    @app.post("/signals/synthesize")
    async def synthesize_signals(conn: AsyncConnection = Depends(_conn_dep)):
        rows = (
            await conn.execute(
                select(Signal)
                .where(Signal.delivered.is_(False))
                .order_by(Signal.created_at)
            )
        ).all()
        if not rows:
            return {"narrative": "No undelivered signals.", "signals": []}

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
            narrative = await signal_synthesizer.synthesize(signal_dicts, context)
        except Exception as exc:
            from loguru import logger
            logger.exception("/signals/synthesize failed")
            raise HTTPException(
                status_code=500, detail=f"synthesis failed: {exc}"
            ) from exc

        return {
            "narrative": narrative,
            "signals": [
                {
                    "id": s["id"],
                    "ticker": s["ticker"],
                    "signal_type": s["signal_type"],
                    "direction": s["direction"],
                    "confidence": s["confidence"],
                }
                for s in signal_dicts
            ],
        }

    return app


async def _conn_dep():
    async with get_connection() as conn:
        yield conn
