"""E-Trade OAuth 1.0a flow via pyetrade.

E-Trade requires an interactive consent step (the user must open a URL in a
browser and paste a verification code back). This module exposes helpers to
walk through that flow and cache the resulting access token tuple for the
session. Persisting tokens across restarts is left to the caller.
"""
from dataclasses import dataclass

import pyetrade
from loguru import logger

from src.config import settings


@dataclass
class ETradeTokens:
    oauth_token: str
    oauth_token_secret: str


_cached: ETradeTokens | None = None


def get_oauth_manager() -> pyetrade.ETradeOAuth:
    return pyetrade.ETradeOAuth(
        settings.ETRADE_CONSUMER_KEY,
        settings.ETRADE_CONSUMER_SECRET,
    )


def authorize_interactive() -> ETradeTokens:
    """Prompt the operator to authorize and return the access token tuple."""
    global _cached
    oauth = get_oauth_manager()
    url = oauth.get_request_token()
    logger.info("Open this URL in a browser to authorize E-Trade: {}", url)
    verifier = input("Paste the verification code: ").strip()
    tokens = oauth.get_access_token(verifier)
    _cached = ETradeTokens(
        oauth_token=tokens["oauth_token"],
        oauth_token_secret=tokens["oauth_token_secret"],
    )
    return _cached


def current_tokens() -> ETradeTokens:
    if _cached is None:
        raise RuntimeError("E-Trade not authorized — call authorize_interactive() first")
    return _cached


def is_sandbox() -> bool:
    return settings.ETRADE_SANDBOX
