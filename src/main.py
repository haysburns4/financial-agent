import asyncio
import sys
from contextlib import asynccontextmanager

import uvicorn
from loguru import logger

from src.backtest import persistence as backtest_persistence
from src.config import settings
from src.db import engine, get_connection, init_db
from src.etrade.auth import auth
from src.llm import health as llm_health
from src.portfolio_history import history_summary
from src.scheduler import build_scheduler, portfolio_history_pipeline
from src.server import create_app


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app):
    _configure_logging()
    await init_db()

    etrade_ready = await auth.load_persisted()
    if etrade_ready:
        logger.info("E-Trade tokens loaded from disk; auth restored")
    else:
        logger.info("No persisted E-Trade tokens; OAuth flow required")

    if backtest_persistence.STORE.load_live_weights():
        meta = backtest_persistence.STORE.metadata
        logger.info(
            "Calibrated signal weights loaded ({} calibration at {})",
            meta["source"], meta["calibrated_at"],
        )
    else:
        logger.info("No calibrated signal weights saved; using the defaults")

    scheduler = build_scheduler()
    scheduler.start()
    logger.info("Scheduler started with {} jobs", len(scheduler.get_jobs()))
    app.state.scheduler = scheduler

    # A local LLM that is down is reported here, not when a chat request hangs.
    llm_summary = await llm_health.log_startup(llm_health.MONITOR)
    logger.info(
        "Ready: E-Trade {}; {}; {}",
        "tokens loaded" if etrade_ready else "needs OAuth",
        llm_summary,
        await _history_banner(),
    )

    # Record snapshot of posiitons on startup
    startup_snapshot: asyncio.Task | None = None
    if settings.SNAPSHOT_ON_STARTUP:
        startup_snapshot = asyncio.create_task(portfolio_history_pipeline.run(source="startup"))

    try:
        yield
    finally:
        if startup_snapshot is not None:
            startup_snapshot.cancel()
        scheduler.shutdown(wait=False)
        await engine.dispose()
        logger.info("Shutdown complete")


async def _history_banner() -> str:
    async with get_connection() as conn:
        summary = await history_summary(conn)
    if summary["latest_snapshot_date"] is None:
        return "portfolio history: none yet"
    return (
        f"portfolio history: latest snapshot {summary['latest_snapshot_date']}, "
        f"{summary['snapshot_dates_recorded']} day(s) recorded"
    )


app = create_app()
app.router.lifespan_context = lifespan


def main() -> None:
    uvicorn.run("src.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)


if __name__ == "__main__":
    main()
