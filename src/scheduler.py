"""APScheduler wiring for the price + portfolio pipelines and signal alerting."""
import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.agent.chat import AgentChat
from src.agent.synthesizer import SignalSynthesizer
from src.alerts.discord import deliver_pending_signals, post_daily_briefing
from src.config import settings
from src.db import engine
from src.etrade.accounts import ETradeAccountClient
from src.etrade.auth import auth
from src.etrade.market import ETradeMarketClient
from src.llm import backend_for
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
# Each task asks for its own backend; provider routing lives in src/llm/factory.py.
signal_synthesizer = SignalSynthesizer(backend_for("synthesizer"))
agent_chat = AgentChat(engine, backend_for("chat"))


async def _price_then_signals() -> None:
    if not await auth.is_authenticated():
        logger.warning("price_pipeline skipped: E-Trade not authenticated")
        return
    tickers = await monitored_tickers(engine)
    logger.info("price_pipeline firing for {} ticker(s): {}", len(tickers), tickers)
    await price_pipeline.run(tickers)
    await signal_engine.run_all(tickers)
    # Per-tick Discord delivery is raw (no synthesizer) so we don't hit
    # Claude every 5 minutes. The synthesizer runs once daily via the
    # `_daily_briefing_job` cron job below, or on demand via /signals/synthesize.
    sent = await deliver_pending_signals()
    if sent:
        logger.info("Discord delivered {} signal(s)", sent)


async def _daily_briefing_run() -> None:
    logger.info("daily briefing firing")
    await post_daily_briefing(signal_synthesizer)


def _daily_briefing_job() -> None:
    try:
        asyncio.run(_daily_briefing_run())
    except Exception:
        logger.exception("daily_briefing raised")


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
    scheduler.add_job(
        _daily_briefing_job,
        trigger=CronTrigger(hour=9, minute=35, timezone=_NYSE_TZ),
        id="daily_briefing",
        name="daily_briefing",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    return scheduler
