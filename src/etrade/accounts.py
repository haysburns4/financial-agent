"""E-Trade accounts wrapper — account list, balances, positions."""
import asyncio
from datetime import datetime, timezone
from typing import Any

import pyetrade
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.etrade.auth import current_tokens, is_sandbox


def _accounts_client() -> pyetrade.ETradeAccounts:
    tokens = current_tokens()
    return pyetrade.ETradeAccounts(
        settings.ETRADE_CONSUMER_KEY,
        settings.ETRADE_CONSUMER_SECRET,
        tokens.oauth_token,
        tokens.oauth_token_secret,
        dev=is_sandbox(),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _list_accounts_sync() -> dict[str, Any]:
    return _accounts_client().list_accounts(resp_format="json")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_portfolio_sync(account_id_key: str) -> dict[str, Any]:
    return _accounts_client().get_account_portfolio(account_id_key, resp_format="json")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _get_balance_sync(account_id_key: str) -> dict[str, Any]:
    return _accounts_client().get_account_balance(account_id_key, resp_format="json")


async def list_accounts() -> list[dict[str, Any]]:
    try:
        payload = await asyncio.to_thread(_list_accounts_sync)
    except Exception:
        logger.exception("E-Trade list_accounts failed")
        return []

    accounts = (
        payload.get("AccountListResponse", {})
        .get("Accounts", {})
        .get("Account", [])
        if isinstance(payload, dict)
        else []
    )
    return accounts if isinstance(accounts, list) else [accounts]


async def fetch_positions() -> list[dict[str, Any]]:
    """Return a flat list of position dicts ready to feed into `PositionIn`."""
    out: list[dict[str, Any]] = []
    accounts = await list_accounts()

    for account in accounts:
        account_id = str(account.get("accountId", ""))
        account_id_key = account.get("accountIdKey")
        if not account_id_key:
            continue

        try:
            payload = await asyncio.to_thread(_get_portfolio_sync, account_id_key)
        except Exception:
            logger.exception("E-Trade get_account_portfolio failed for {}", account_id)
            continue

        portfolios = (
            payload.get("PortfolioResponse", {}).get("AccountPortfolio", [])
            if isinstance(payload, dict)
            else []
        )
        if isinstance(portfolios, dict):
            portfolios = [portfolios]

        now = datetime.now(timezone.utc)
        for portfolio in portfolios:
            positions = portfolio.get("Position", [])
            if isinstance(positions, dict):
                positions = [positions]
            for p in positions:
                product = p.get("Product", {}) if isinstance(p, dict) else {}
                ticker = product.get("symbol")
                if not ticker:
                    continue
                out.append(
                    {
                        "account_id": account_id,
                        "ticker": ticker,
                        "quantity": float(p.get("quantity", 0.0)),
                        "cost_basis": float(p.get("costPerShare", 0.0)) * float(p.get("quantity", 0.0)),
                        "market_value": float(p.get("marketValue", 0.0)),
                        "last_updated": now,
                    }
                )

    return out
