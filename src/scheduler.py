import asyncio

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.config import settings
from src.pipelines.portfolio_pipeline import run_portfolio_pipeline
from src.pipelines.price_pipeline import run_price_pipeline


def _run_async(coro_factory):
    """Bridge APScheduler's sync worker thread into asyncio."""
    def _job():
        try:
            asyncio.run(coro_factory())
        except Exception:
            logger.exception("Scheduled job raised")
    return _job


def build_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        _run_async(run_price_pipeline),
        trigger=IntervalTrigger(minutes=settings.PRICE_POLL_MINUTES),
        id="price_pipeline",
        name="Fetch OHLCV for watchlist",
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        _run_async(run_portfolio_pipeline),
        trigger=IntervalTrigger(minutes=settings.PORTFOLIO_POLL_MINUTES),
        id="portfolio_pipeline",
        name="Fetch positions and balances",
        max_instances=1,
        coalesce=True,
    )

    return scheduler
