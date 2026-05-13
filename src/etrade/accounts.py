"""E-Trade accounts client — account list, positions, balances."""
import asyncio
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


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class ETradeAccountClient:
    def __init__(self, auth: ETradeAuth) -> None:
        self._auth = auth

    async def list_accounts(self) -> list[dict]:
        try:
            session = await self._auth.get_accounts_session()
            payload = await asyncio.to_thread(self._list_accounts_sync, session)
        except Exception:
            logger.exception("E-Trade list_accounts failed")
            return []

        accounts = (
            payload.get("AccountListResponse", {}).get("Accounts", {}).get("Account", [])
            if isinstance(payload, dict)
            else []
        )
        accounts = _as_list(accounts)

        out: list[dict] = []
        for a in accounts:
            if not isinstance(a, dict):
                continue
            out.append(
                {
                    "account_id": str(a.get("accountId", "")),
                    "account_id_key": a.get("accountIdKey"),
                    "description": a.get("accountDesc") or a.get("institutionType") or "",
                }
            )
        return out

    async def get_positions(self, account_id: str) -> list[dict]:
        try:
            session = await self._auth.get_accounts_session()
            payload = await asyncio.to_thread(self._get_portfolio_sync, session, account_id)
        except Exception:
            logger.exception("E-Trade get_positions failed for {}", account_id)
            return []

        portfolios = (
            payload.get("PortfolioResponse", {}).get("AccountPortfolio", [])
            if isinstance(payload, dict)
            else []
        )
        portfolios = _as_list(portfolios)

        out: list[dict] = []
        for portfolio in portfolios:
            if not isinstance(portfolio, dict):
                continue
            for p in _as_list(portfolio.get("Position")):
                if not isinstance(p, dict):
                    continue
                try:
                    product = p.get("Product") if isinstance(p.get("Product"), dict) else {}
                    ticker = product.get("symbol")
                    if not ticker:
                        logger.warning("E-Trade position missing symbol; skipping")
                        continue
                    quantity = _to_float(p.get("quantity"))
                    out.append(
                        {
                            "ticker": ticker,
                            "quantity": quantity,
                            "costBasis": _to_float(p.get("costPerShare")) * quantity,
                            "marketValue": _to_float(p.get("marketValue")),
                            "pctGain": _to_float(p.get("totalGainPct")),
                        }
                    )
                except Exception:
                    logger.exception("E-Trade position parse failed; skipping entry")
                    continue

        return out

    async def get_balance(self, account_id: str) -> dict:
        try:
            session = await self._auth.get_accounts_session()
            payload = await asyncio.to_thread(self._get_balance_sync, session, account_id)
        except Exception:
            logger.exception("E-Trade get_balance failed for {}", account_id)
            return {"cash_balance": 0.0, "total_market_value": 0.0, "day_gain_loss": 0.0}

        computed = (
            payload.get("BalanceResponse", {}).get("Computed", {})
            if isinstance(payload, dict)
            else {}
        )
        if not isinstance(computed, dict):
            computed = {}
        real_time = computed.get("RealTimeValues") if isinstance(computed.get("RealTimeValues"), dict) else {}

        return {
            "cash_balance": _to_float(computed.get("cashAvailableForInvestment")),
            "total_market_value": _to_float(real_time.get("totalAccountValue")),
            "day_gain_loss": _to_float(real_time.get("totalDayGainLoss")),
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _list_accounts_sync(self, session) -> dict:
        return session.list_accounts(resp_format="json")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _get_portfolio_sync(self, session, account_id_key: str) -> dict:
        return session.get_account_portfolio(account_id_key, resp_format="json")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _get_balance_sync(self, session, account_id_key: str) -> dict:
        return session.get_account_balance(account_id_key, resp_format="json")
