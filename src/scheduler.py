"""APScheduler wiring for the price + portfolio pipelines and signal alerting."""
import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.alerts.discord import deliver_pending_signals
from src.config import settings
from src.db import engine
from src.etrade.accounts import ETradeAccountClient
from src.etrade.auth import auth
from src.etrade.market import ETradeMarketClient
from src.pipelines import monitored_tickers
from src.pipelines.portfolio_pipeline import PortfolioPipeline
from src.pipelines.price_pipeline import PricePipeline
from src.signals.engine import SignalEngine


_NYSE_TZ = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def market_is_open(now: datetime | None = None) -> bool:
    """Simple NYSE check: weekday, 9:30am-4:00pm ET.

    Ignores holidays — adequate for sandbox-mode polling. Swap for
    pandas_market_calendars if exact NYSE calendar is needed.
    """
    moment = (now or datetime.now(tz=_NYSE_TZ)).astimezone(_NYSE_TZ)
    if moment.weekday() >= 5:
        return False
    return _MARKET_OPEN <= moment.time() < _MARKET_CLOSE


# Singletons shared between the scheduler and the FastAPI endpoints so that
# circuit-breaker state accumulates consistently.
market_client = ETradeMarketClient(auth)
account_client = ETradeAccountClient(auth)
price_pipeline = PricePipeline(market_client, engine)
portfolio_pipeline = PortfolioPipeline(account_client, engine)
signal_engine = SignalEngine(engine)


async def _price_then_signals() -> None:
    if not await auth.is_authenticated():
        logger.warning("price_pipeline skipped: E-Trade not authenticated")
        return
    tickers = await monitored_tickers(engine)
    logger.info("price_pipeline firing for {} ticker(s): {}", len(tickers), tickers)
    await price_pipeline.run(tickers)
    await signal_engine.run_all(tickers)
    sent = await deliver_pending_signals()
    if sent:
        logger.info("Discord delivered {} signal(s)", sent)


def _price_job() -> None:
    if not market_is_open():
        logger.info("price_pipeline skipped: market closed")
        return
    try:
        asyncio.run(_price_then_signals())
    except Exception:
        logger.exception("price_pipeline chain raised")


async def _portfolio_run() -> None:
    if not await auth.is_authenticated():
        logger.warning("portfolio_pipeline skipped: E-Trade not authenticated")
        return
    logger.info("portfolio_pipeline firing")
    await portfolio_pipeline.run()


def _portfolio_job() -> None:
    try:
        asyncio.run(_portfolio_run())
    except Exception:
        logger.exception("portfolio_pipeline raised")


def build_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        _price_job,
        trigger=IntervalTrigger(minutes=settings.PRICE_POLL_MINUTES),
        id="price_pipeline",
        name="price_pipeline",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )
    scheduler.add_job(
        _portfolio_job,
        trigger=IntervalTrigger(minutes=settings.PORTFOLIO_POLL_MINUTES),
        id="portfolio_pipeline",
        name="portfolio_pipeline",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )
    return scheduler
