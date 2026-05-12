"""APScheduler wiring for the price pipeline."""
import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.config import settings
from src.db import engine
from src.etrade.auth import auth
from src.etrade.market import ETradeMarketClient
from src.pipelines.price_pipeline import PricePipeline


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


# Singletons that survive across scheduler firings so circuit-breaker state
# accumulates correctly and so the manual /pipeline/price/run endpoint shares
# them with the scheduler.
market_client = ETradeMarketClient(auth)
price_pipeline = PricePipeline(market_client, engine)


def _price_job() -> None:
    if not market_is_open():
        logger.info("price_pipeline skipped: market closed")
        return
    if not auth.is_authenticated():
        logger.warning("price_pipeline skipped: E-Trade not authenticated")
        return
    logger.info("price_pipeline firing for {} tickers", len(settings.WATCHLIST))
    try:
        asyncio.run(price_pipeline.run(settings.WATCHLIST))
    except Exception:
        logger.exception("price_pipeline run raised")


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
    return scheduler
