"""E-Trade OAuth 1.0a flow with persistent, encrypted token storage.

The OAuth flow is driven over HTTP: `start_auth()` returns an authorize URL,
the user pastes a verifier into `complete_auth(verifier)`, and the resulting
tokens are encrypted and written to the local SQLite DB. On process restart,
`load_persisted()` decrypts and rehydrates the singleton so the user doesn't
have to re-authenticate.

Tokens are opportunistically renewed via `pyetrade.ETradeAccessManager` once
they cross `RENEW_THRESHOLD_MIN` minutes of age — just under E-Trade's 2-hour
idle TTL. Renewal failure (typically the midnight-ET daily invalidation)
clears the tokens and forces re-auth.

Persistence (encrypt/decrypt + DB I/O) lives in [src/etrade/token_store.py];
this module only owns the in-memory state and the OAuth/renewal lifecycle.
"""
import asyncio
import re
from collections.abc import Callable
from datetime import datetime, timezone

import pyetrade
from loguru import logger
from requests_oauthlib.oauth1_session import TokenRequestDenied

from src.config import settings
from src.etrade import token_store
from src.etrade.token_store import ETradeTokens  # re-exported for callers


class ETradeAuthError(Exception):
    """E-Trade refused the OAuth handshake.

    Raised instead of leaking `TokenRequestDenied` and a page of HTML;
    `oauth_problem` carries E-Trade's reason, e.g. `consumer_key_rejected`.
    """

    def __init__(self, message: str, oauth_problem: str | None = None) -> None:
        super().__init__(message)
        self.oauth_problem = oauth_problem


def _denied(exc: TokenRequestDenied, stage: str) -> ETradeAuthError:
    # Don't touch exc.status_code — it dereferences exc.response, which is
    # optional. The reason we want is in the body E-Trade echoed back.
    match = re.search(r"oauth_problem=([A-Za-z_]+)", str(exc))
    problem = match.group(1) if match else None
    return ETradeAuthError(
        f"E-Trade rejected the {stage} request: {problem or str(exc)[:200]}",
        problem,
    )


class ETradeAuth:
    RENEW_THRESHOLD_MIN = 110

    def __init__(
        self,
        oauth_factory: Callable[[str, str], pyetrade.ETradeOAuth] = pyetrade.ETradeOAuth,
    ) -> None:
        # Injected so tests can supply a stand-in without patching an import path.
        self._oauth_factory = oauth_factory
        self._oauth: pyetrade.ETradeOAuth | None = None
        self._tokens: ETradeTokens | None = None

    # ---------- OAuth flow ----------

    def start_auth(self) -> str:
        self._oauth = self._oauth_factory(
            settings.ETRADE_CONSUMER_KEY,
            settings.ETRADE_CONSUMER_SECRET,
        )
        try:
            url = self._oauth.get_request_token()
        except TokenRequestDenied as exc:
            raise _denied(exc, "request token") from exc
        logger.info("E-Trade auth started; visit URL to obtain verifier: {}", url)
        return url

    async def complete_auth(self, verifier: str) -> None:
        if self._oauth is None:
            raise RuntimeError("complete_auth() called before start_auth()")
        try:
            tokens = self._oauth.get_access_token(verifier)
        except TokenRequestDenied as exc:
            raise _denied(exc, "access token") from exc
        now = datetime.now(timezone.utc)
        self._tokens = ETradeTokens(
            oauth_token=tokens["oauth_token"],
            oauth_token_secret=tokens["oauth_token_secret"],
            authenticated_at=now,
        )
        logger.info("E-Trade authenticated at {}", now.isoformat())
        await self.persist_tokens()

    # ---------- auth checks ----------

    async def is_authenticated(self) -> bool:
        if self._tokens is None:
            return False
        if self._needs_renewal():
            return await self._maybe_renew()
        return True

    def session_age_minutes(self) -> int | None:
        if self._tokens is None:
            return None
        age = datetime.now(timezone.utc) - self._tokens.authenticated_at
        return int(age.total_seconds() // 60)

    async def get_market_session(self) -> pyetrade.ETradeMarket:
        tokens = await self._require_fresh_tokens()
        return pyetrade.ETradeMarket(
            settings.ETRADE_CONSUMER_KEY,
            settings.ETRADE_CONSUMER_SECRET,
            tokens.oauth_token,
            tokens.oauth_token_secret,
            dev=settings.ETRADE_SANDBOX,
        )

    async def get_accounts_session(self) -> pyetrade.ETradeAccounts:
        tokens = await self._require_fresh_tokens()
        return pyetrade.ETradeAccounts(
            settings.ETRADE_CONSUMER_KEY,
            settings.ETRADE_CONSUMER_SECRET,
            tokens.oauth_token,
            tokens.oauth_token_secret,
            dev=settings.ETRADE_SANDBOX,
        )

    async def _require_fresh_tokens(self) -> ETradeTokens:
        if self._tokens is None:
            raise RuntimeError("E-Trade not authenticated — call start_auth() / complete_auth() first")
        if self._needs_renewal() and not await self._maybe_renew():
            raise RuntimeError("E-Trade tokens expired and renewal failed — re-authenticate")
        return self._tokens

    def _needs_renewal(self) -> bool:
        age_minutes = self.session_age_minutes()
        return age_minutes is not None and age_minutes >= self.RENEW_THRESHOLD_MIN

    async def _maybe_renew(self) -> bool:
        if self._tokens is None:
            return False
        manager = pyetrade.ETradeAccessManager(
            settings.ETRADE_CONSUMER_KEY,
            settings.ETRADE_CONSUMER_SECRET,
            self._tokens.oauth_token,
            self._tokens.oauth_token_secret,
        )
        try:
            ok = await asyncio.to_thread(manager.renew_access_token)
        except Exception as exc:
            logger.warning("E-Trade renewal failed ({}); clearing tokens", exc)
            await self.clear_persisted()
            return False
        if not ok:
            logger.warning("E-Trade renewal returned False; clearing tokens")
            await self.clear_persisted()
            return False
        self._tokens.authenticated_at = datetime.now(timezone.utc)
        logger.info("E-Trade tokens renewed")
        await self.persist_tokens()
        return True

    # ---------- persistence glue ----------

    async def persist_tokens(self) -> bool:
        if self._tokens is None:
            return False
        return await token_store.save(self._tokens)

    async def load_persisted(self) -> bool:
        tokens = await token_store.load()
        if tokens is None:
            return False
        self._tokens = tokens
        logger.info(
            "Loaded persisted E-Trade tokens (authenticated {}m ago)",
            self.session_age_minutes(),
        )
        return True

    async def clear_persisted(self) -> None:
        self._tokens = None
        self._oauth = None
        await token_store.clear()


auth = ETradeAuth()
