"""E-Trade market data client.

`pyetrade` is synchronous; we run its calls in a worker thread so the asyncio
event loop isn't blocked. Network-level failures are retried via tenacity.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.etrade.auth import ETradeAuth


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ETradeMarketClient:
    def __init__(self, auth: ETradeAuth) -> None:
        self._auth = auth

    async def get_quotes(self, tickers: list[str]) -> list[dict]:
        if not tickers:
            return []
        try:
            payload = await asyncio.to_thread(self._get_quotes_sync, tickers)
        except Exception:
            logger.exception("E-Trade get_quotes failed for {}", tickers)
            return []

        quote_data = (
            payload.get("QuoteResponse", {}).get("QuoteData", [])
            if isinstance(payload, dict)
            else []
        )
        if isinstance(quote_data, dict):
            quote_data = [quote_data]

        now = datetime.now(timezone.utc)
        results: list[dict] = []
        for q in quote_data:
            if not isinstance(q, dict):
                continue

            messages = q.get("Messages", {}) or {}
            msg_list = messages.get("Message", []) if isinstance(messages, dict) else []
            if isinstance(msg_list, dict):
                msg_list = [msg_list]
            error_msgs = [
                m for m in msg_list
                if isinstance(m, dict) and str(m.get("type", "")).upper() in ("WARNING", "ERROR")
            ]
            if error_msgs:
                logger.warning(
                    "E-Trade quote returned messages for {}: {}",
                    q.get("Product", {}).get("symbol"),
                    error_msgs,
                )

            all_data = q.get("All")
            if not isinstance(all_data, dict):
                logger.warning(
                    "E-Trade quote missing 'All' block for {}; skipping",
                    q.get("Product", {}).get("symbol"),
                )
                continue

            ticker = q.get("Product", {}).get("symbol") if isinstance(q.get("Product"), dict) else None
            if not ticker:
                logger.warning("E-Trade quote missing symbol; skipping entry")
                continue

            results.append(
                {
                    "ticker": ticker,
                    "last": _to_float(all_data.get("lastTrade")),
                    "bid": _to_float(all_data.get("bid")),
                    "ask": _to_float(all_data.get("ask")),
                    "high": _to_float(all_data.get("high")),
                    "low": _to_float(all_data.get("low")),
                    "open": _to_float(all_data.get("open")),
                    "close": _to_float(all_data.get("previousClose")),
                    "volume": _to_float(all_data.get("totalVolume")),
                    "timestamp": now,
                }
            )

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _get_quotes_sync(self, tickers: list[str]) -> dict:
        return self._auth.get_market_session().get_quote(tickers, resp_format="json")
