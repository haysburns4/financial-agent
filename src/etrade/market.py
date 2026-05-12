"""E-Trade market data wrapper.

`pyetrade` is a synchronous library; we run its calls in a worker thread so
the asyncio scheduler isn't blocked.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any

import pyetrade
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.etrade.auth import current_tokens, is_sandbox


def _market_client() -> pyetrade.ETradeMarket:
    tokens = current_tokens()
    return pyetrade.ETradeMarket(
        settings.ETRADE_CONSUMER_KEY,
        settings.ETRADE_CONSUMER_SECRET,
        tokens.oauth_token,
        tokens.oauth_token_secret,
        dev=is_sandbox(),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_quote_sync(ticker: str) -> dict[str, Any]:
    return _market_client().get_quote([ticker], resp_format="json")


async def fetch_ohlcv(ticker: str) -> list[dict[str, Any]]:
    """Return a list of OHLCV bar dicts ready to feed into `PriceBarIn`.

    E-Trade's quote endpoint returns a single snapshot, not a series. For now
    we materialize a one-bar series from that snapshot; swap this out for the
    historical-bars endpoint when ready.
    """
    try:
        payload = await asyncio.to_thread(_get_quote_sync, ticker)
    except Exception:
        logger.exception("E-Trade quote failed for {}", ticker)
        return []

    quotes = (
        payload.get("QuoteResponse", {}).get("QuoteData", [])
        if isinstance(payload, dict)
        else []
    )
    bars: list[dict[str, Any]] = []
    for q in quotes:
        all_data = q.get("All", {}) if isinstance(q, dict) else {}
        bars.append(
            {
                "ticker": ticker,
                "timestamp": datetime.now(timezone.utc),
                "open": float(all_data.get("open", 0.0)),
                "high": float(all_data.get("high", 0.0)),
                "low": float(all_data.get("low", 0.0)),
                "close": float(all_data.get("lastTrade", 0.0)),
                "volume": float(all_data.get("totalVolume", 0.0)),
                "adjusted_close": float(all_data.get("lastTrade", 0.0)),
                "data_quality": "ok",
            }
        )
    return bars
