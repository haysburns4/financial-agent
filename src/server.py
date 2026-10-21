from dataclasses import asdict
from datetime import date, timezone
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import and_, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection

from src.agent.archive import archive_summary, get_output, list_outputs
from src.agui import create_agui_router
from src.backtest import persistence as backtest_persistence
from src.backtest.calibration import calibrate
from src.signals.catchup import catch_up
from src.backtest.runner import BacktestRunner
from src.config import settings
from src.db import engine, get_connection, health_check
from src.etrade.auth import ETradeAuthError, auth
from src.etrade.client import order_endpoints_blocked
from src.llm import health as llm_health
from src.models import Indicator, PipelineRun, Position, PriceBar, Signal, WatchlistItem
from src.portfolio import dashboard, position_dict, risk_by_account, risk_summary
from src.portfolio_history import coverage, history_rows, history_summary
from src.pipelines.backfill_pipeline import BackfillPipeline
from src.pipelines.portfolio_history_pipeline import SnapshotRunResult
from src.scheduler import (
    LLM_REPLY_TOKENS,
    agent_chat,
    portfolio_history_pipeline,
    portfolio_pipeline,
    price_pipeline,
    signal_synthesizer,
    watchlist_service,
)
from src.signals.engine import FILTER_STATS, SIGNAL_CATEGORIES, signal_types_for_category
from src.watchlist.resolver import get_monitored, monitored_on
from src.watchlist.resolver import tickers as monitored_tickers
from src.watchlist.service import UNSET, WatchlistError, active_count, list_events


class CompleteAuthBody(BaseModel):
    verifier: str


class SnapshotRunRequest(BaseModel):
    # Pull positions from E-Trade first if they are over 30 minutes old.
    refresh_first: bool = True


class BackfillRequest(BaseModel):
    tickers: list[str] | None = None
    period_daily: str = "2y"
    period_intraday: str = "60d"
    interval_intraday: str = "5m"
    # Re-run the walk-forward over all stored history and apply it afterwards.
    calibrate: bool = True
    # Then record the signals missed since the last check (src/signals/catchup.py).
    catch_up: bool = True


class BacktestRequest(BaseModel):
    tickers: list[str] | None = None
    start_date: date
    end_date: date
    forward_window_days: int = 5
    outcome_threshold_pct: float = 0.01
    # "daily": years of history; "intraday": the 5-minute bars live signals
    # fire on (only as far back as backfills have collected them).
    timeframe: Literal["daily", "intraday"] = "daily"


class WalkforwardRequest(BacktestRequest):
    train_window_months: int = 12
    test_window_months: int = 1
    step_months: int = 1


class WatchlistAdd(BaseModel):
    ticker: str
    note: str | None = None
    tags: list[str] | None = None
    # The web UI sends "ui"; everything else is an API call.
    source: Literal["ui", "api"] = "api"


class WatchlistPatch(BaseModel):
    # Omitted fields are left alone; `"note": null` clears the note.
    note: str | None = None
    tags: list[str] | None = None
    source: Literal["ui", "api"] = "api"


# How each service rejection reads over HTTP.
_WATCHLIST_STATUS = {
    "invalid_ticker": 422,
    "invalid_tag": 422,
    "unknown_symbol": 422,
    "at_capacity": 409,
    "not_failed": 409,
    "not_found": 404,
}


def _watchlist_error(exc: WatchlistError) -> HTTPException:
    return HTTPException(status_code=_WATCHLIST_STATUS[exc.code], detail=str(exc))


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_history: list[ChatTurn] | None = None
    # The highlighted account, analyzed by default; omit for every account.
    account_id: str | None = None
    # From a previous response, to thread this turn into that conversation.
    conversation_id: str | None = None
    # The dashboard's Watchlist tab is open: signals cover watchlist tickers only.
    watchlist_only: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="financial-agent", version="0.1.0")
    app.include_router(create_agui_router(agent_chat))

    @app.get("/health")
    async def health():
        ok = await health_check()
        llm, alerts = await llm_health.llm_health(llm_health.MONITOR, reply_tokens=LLM_REPLY_TOKENS)
        history = archive = watchlist = None
        if ok:
            try:
                async with get_connection() as conn:
                    history = await history_summary(conn)
                    archive = await archive_summary(conn)
                    watchlist = await _watchlist_health(conn)
            except SQLAlchemyError:
                logger.exception("health: portfolio history, agent archive or watchlist unavailable")
        # `service` lets the launcher recognise an instance it can reuse.
        # A down local LLM degrades the service: market data still collects.
        return {
            "status": "ok" if ok and not alerts else "degraded",
            "db": ok,
            "llm": llm,
            "etrade": {
                "authenticated": await auth.is_authenticated(),
                "sandbox": settings.ETRADE_SANDBOX,
                "session_age_minutes": auth.session_age_minutes(),
                "mode": "read_only",
                "order_endpoints_blocked": order_endpoints_blocked(),
            },
            "alerts": alerts,
            "portfolio_history": history,
            "agent_archive": archive,
            "watchlist": watchlist,
            "service": "financial-agent",
            "signal_filter_stats": {"last_24h": FILTER_STATS.last_24h()},
        }

    @app.post("/auth/start")
    async def auth_start():
        try:
            url = auth.start_auth()
        except ETradeAuthError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"auth_url": url}

    @app.post("/auth/complete")
    async def auth_complete(body: CompleteAuthBody, background: BackgroundTasks):
        try:
            await auth.complete_auth(body.verifier)
        except ETradeAuthError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        # Startup usually comes before the (interactive) login, so its snapshot
        # is missing or copied stale positions. Capture again.
        if settings.SNAPSHOT_ON_STARTUP and await portfolio_history_pipeline.needs_capture():
            background.add_task(portfolio_history_pipeline.run, source="startup")
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
            ticker_list = monitored_tickers(await get_monitored(engine))
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

    @app.get("/watchlist")
    async def watchlist_active():
        return [e.as_json() for e in await watchlist_service.list_active()]

    @app.get("/watchlist/monitored")
    async def watchlist_monitored():
        return [asdict(m) for m in await get_monitored(engine)]

    @app.get("/watchlist/events")
    async def watchlist_events(
        ticker: str | None = None,
        limit: int = Query(100, ge=1, le=1000),
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        try:
            return await list_events(conn, ticker, limit)
        except WatchlistError as exc:
            raise _watchlist_error(exc) from exc

    @app.post("/watchlist", status_code=201)
    async def watchlist_add(body: WatchlistAdd):
        try:
            entry = await watchlist_service.add(body.ticker, body.note, body.tags, source=body.source)
        except WatchlistError as exc:
            raise _watchlist_error(exc) from exc
        return entry.as_json()

    @app.patch("/watchlist/{ticker}")  # anti-slop: allow no-module-mocking - FastAPI PATCH route, not mock.patch
    async def watchlist_patch(ticker: str, body: WatchlistPatch):
        given = body.model_fields_set
        try:
            entry = await watchlist_service.update(
                ticker,
                note=body.note if "note" in given else UNSET,
                tags=(body.tags or []) if "tags" in given else UNSET,
                source=body.source,
            )
        except WatchlistError as exc:
            raise _watchlist_error(exc) from exc
        return entry.as_json()

    @app.delete("/watchlist/{ticker}")
    async def watchlist_remove(ticker: str, source: Literal["ui", "api"] = "api"):
        try:
            result = await watchlist_service.remove(ticker, source=source)
        except WatchlistError as exc:
            raise _watchlist_error(exc) from exc
        return {**result.entry.as_json(), "held": result.held}

    @app.post("/watchlist/{ticker}/backfill", status_code=202)
    async def watchlist_retry_backfill(ticker: str):
        try:
            entry = await watchlist_service.retry_backfill(ticker)
        except WatchlistError as exc:
            raise _watchlist_error(exc) from exc
        return entry.as_json()

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

    @app.get("/agent/history")
    async def agent_history(
        type: Literal["synthesis", "chat"] | None = None,
        conversation_id: str | None = None,
        ticker: str | None = None,
        start: date | None = None,
        end: date | None = None,
        limit: int = Query(100, ge=1, le=1000),
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        """Archived syntheses and chat turns, newest first; `start`/`end` are
        inclusive UTC dates."""
        return await list_outputs(
            conn, output_type=type, conversation_id=conversation_id, ticker=ticker,
            start=start, end=end, limit=limit,
        )

    @app.get("/agent/history/{output_id:int}")
    async def agent_history_item(output_id: int, conn: AsyncConnection = Depends(_conn_dep)):
        output = await get_output(conn, output_id)
        if output is None:
            raise HTTPException(status_code=404, detail=f"no agent output {output_id}")
        return output

    @app.get("/portfolio/history")
    async def portfolio_history(
        account_id: str | None = None,
        ticker: str | None = None,
        start: date | None = None,
        end: date | None = None,
        limit: int = Query(500, ge=1, le=10_000),
        conn: AsyncConnection = Depends(_conn_dep),
    ):
        """Snapshot rows, newest trading day first: a flat list for one account,
        else grouped by account. `limit` caps the rows before grouping."""
        rows = await history_rows(
            conn,
            account_id=account_id,
            ticker=ticker.strip().upper() if ticker else None,
            start=start,
            end=end,
            limit=limit,
        )
        if account_id is not None:
            return rows
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["account_id"], []).append(row)
        return grouped

    @app.get("/portfolio/history/coverage")
    async def portfolio_history_coverage(conn: AsyncConnection = Depends(_conn_dep)):
        return await coverage(conn)

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

    @app.post("/pipeline/portfolio_history/run")
    async def trigger_portfolio_history_run(body: SnapshotRunRequest | None = None):
        # Never raises: an E-Trade or DB problem is reported in `errors`.
        result = await portfolio_history_pipeline.run(
            source="manual", refresh_first=(body or SnapshotRunRequest()).refresh_first
        )
        return _snapshot_result(result)

    @app.post("/pipeline/backfill/run")
    async def trigger_backfill(body: BackfillRequest | None = None):
        body = body or BackfillRequest()
        tickers = body.tickers or monitored_tickers(await get_monitored(engine))
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
        # Fresh data -> fresh confidence weights. A calibration problem never
        # fails the backfill: the bars are stored either way.
        calibration: dict | None = None
        if body.calibrate:
            try:
                calibration = await calibrate(engine, tickers)
            except ValueError as exc:
                calibration = {"not_run": str(exc)}
            except Exception as exc:
                logger.exception("calibration after backfill failed")
                calibration = {"error": f"{type(exc).__name__}: {exc}"}
        # After calibration, so caught-up signals carry the fresh confidences.
        signals: dict | None = None
        if body.catch_up:
            try:
                signals = asdict(await catch_up(engine, tickers))
            except Exception as exc:
                logger.exception("signal catch-up after backfill failed")
                signals = {"error": f"{type(exc).__name__}: {exc}"}
        return {**asdict(result), "calibration": calibration, "signals": signals}

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
        tickers = body.tickers or monitored_tickers(await get_monitored(engine))
        runner = BacktestRunner(engine)
        report = await runner.run(
            tickers=tickers,
            start_date=body.start_date,
            end_date=body.end_date,
            forward_window_days=body.forward_window_days,
            outcome_threshold_pct=body.outcome_threshold_pct,
            timeframe=body.timeframe,
        )
        backtest_persistence.STORE.save_report(report)
        return asdict(report)

    @app.get("/backtest/latest")
    async def latest_backtest():
        report = backtest_persistence.STORE.load_report_json()
        if report is None:
            raise HTTPException(status_code=404, detail="no backtest report on disk")
        return report

    @app.post("/backtest/walkforward/run")
    async def trigger_walkforward(body: WalkforwardRequest):
        tickers = body.tickers or monitored_tickers(await get_monitored(engine))
        try:
            report = await BacktestRunner(engine).run_walkforward(
                tickers=tickers,
                start_date=body.start_date,
                end_date=body.end_date,
                train_window_months=body.train_window_months,
                test_window_months=body.test_window_months,
                step_months=body.step_months,
                forward_window_days=body.forward_window_days,
                outcome_threshold_pct=body.outcome_threshold_pct,
                timeframe=body.timeframe,
            )
        except ValueError as exc:
            # Bad windowing (e.g. range shorter than one train + test span).
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        backtest_persistence.STORE.save_walkforward_report(report)
        return asdict(report)

    @app.get("/backtest/walkforward/latest")
    async def latest_walkforward():
        report = backtest_persistence.STORE.load_walkforward_json()
        if report is None:
            raise HTTPException(status_code=404, detail="no walk-forward report on disk")
        return report

    @app.post("/backtest/apply")
    async def apply_backtest():
        """Calibrate from the newer of the two reports; `source` says which."""
        try:
            return backtest_persistence.STORE.apply_latest_report()
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
            return await agent_chat.ask(
                body.question, history, body.account_id, body.conversation_id, body.watchlist_only
            )
        except Exception as exc:
            from loguru import logger
            logger.exception("/chat failed")
            raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

    @app.post("/signals/synthesize")
    async def synthesize_signals(hours: int = Query(24, ge=1, le=24 * 30)):
        """A briefing on the signals of the last `hours`, archived like the
        daily one."""
        try:
            briefing = await signal_synthesizer.brief_recent(hours)
        except Exception as exc:
            from loguru import logger
            logger.exception("/signals/synthesize failed")
            raise HTTPException(
                status_code=500, detail=f"synthesis failed: {exc}"
            ) from exc
        if briefing is None:
            return {"narrative": f"No signals in the last {hours}h.", "signals": []}

        return {
            "narrative": briefing.narrative,
            "signals": [
                {
                    "id": s["id"],
                    "ticker": s["ticker"],
                    "signal_type": s["signal_type"],
                    "direction": s["direction"],
                    "confidence": s["confidence"],
                }
                for s in briefing.signals
            ],
        }

    return app


async def _watchlist_health(conn: AsyncConnection) -> dict:
    monitored = await monitored_on(conn)
    rows = (
        await conn.execute(
            select(WatchlistItem.ticker, WatchlistItem.backfill_status)
            .where(WatchlistItem.removed_at.is_(None), WatchlistItem.backfill_status != "done")
            .order_by(WatchlistItem.ticker)
        )
    ).all()
    backfill: dict[str, list[str]] = {"pending": [], "running": [], "failed": []}
    for r in rows:
        backfill[r.backfill_status].append(r.ticker)
    return {
        "active": await active_count(conn),
        "monitored": len(monitored),
        "backfill": backfill,
    }


def _snapshot_result(result: SnapshotRunResult) -> dict:
    return {
        **asdict(result),
        "snapshot_date": result.snapshot_date.isoformat(),
        "captured_at": result.captured_at.isoformat(),
    }


async def _conn_dep():
    async with get_connection() as conn:
        yield conn
