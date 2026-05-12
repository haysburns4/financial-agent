"""E-Trade OAuth 1.0a flow via pyetrade.

The OAuth flow is driven over HTTP: clients call `start_auth()` to get the
E-Trade authorization URL, the user visits it in a browser to obtain a
verification code, then `complete_auth(verifier)` exchanges that for access
tokens. Tokens live on the singleton `auth` instance for the life of the
process — they are not persisted across restarts.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pyetrade
from loguru import logger

from src.config import settings


@dataclass
class ETradeTokens:
    oauth_token: str
    oauth_token_secret: str
    authenticated_at: datetime


class ETradeAuth:
    def __init__(self) -> None:
        self._oauth: pyetrade.ETradeOAuth | None = None
        self._tokens: ETradeTokens | None = None

    def start_auth(self) -> str:
        self._oauth = pyetrade.ETradeOAuth(
            settings.ETRADE_CONSUMER_KEY,
            settings.ETRADE_CONSUMER_SECRET,
        )
        url = self._oauth.get_request_token()
        logger.info("E-Trade auth started; visit URL to obtain verifier: {}", url)
        return url

    def complete_auth(self, verifier: str) -> None:
        if self._oauth is None:
            raise RuntimeError("complete_auth() called before start_auth()")
        tokens = self._oauth.get_access_token(verifier)
        now = datetime.now(timezone.utc)
        self._tokens = ETradeTokens(
            oauth_token=tokens["oauth_token"],
            oauth_token_secret=tokens["oauth_token_secret"],
            authenticated_at=now,
        )
        logger.info("E-Trade authenticated at {}", now.isoformat())

    def is_authenticated(self) -> bool:
        if self._tokens is None:
            return False
        age = datetime.now(timezone.utc) - self._tokens.authenticated_at
        if age > timedelta(hours=2):
            logger.warning(
                "E-Trade session is {} old; tokens may be expired", age
            )
        return True

    def session_age_minutes(self) -> int | None:
        if self._tokens is None:
            return None
        age = datetime.now(timezone.utc) - self._tokens.authenticated_at
        return int(age.total_seconds() // 60)

    def get_market_session(self) -> pyetrade.ETradeMarket:
        tokens = self._require_tokens()
        return pyetrade.ETradeMarket(
            settings.ETRADE_CONSUMER_KEY,
            settings.ETRADE_CONSUMER_SECRET,
            tokens.oauth_token,
            tokens.oauth_token_secret,
            dev=settings.ETRADE_SANDBOX,
        )

    def get_accounts_session(self) -> pyetrade.ETradeAccounts:
        tokens = self._require_tokens()
        return pyetrade.ETradeAccounts(
            settings.ETRADE_CONSUMER_KEY,
            settings.ETRADE_CONSUMER_SECRET,
            tokens.oauth_token,
            tokens.oauth_token_secret,
            dev=settings.ETRADE_SANDBOX,
        )

    def _require_tokens(self) -> ETradeTokens:
        if self._tokens is None:
            raise RuntimeError("E-Trade not authenticated — call start_auth() / complete_auth() first")
        return self._tokens


auth = ETradeAuth()
