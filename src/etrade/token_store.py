"""Persistence layer for E-Trade OAuth tokens.

Tokens are encrypted with Fernet (key from `settings.TOKEN_ENCRYPTION_KEY`)
and stored in the singleton `etrade_credentials` row (id=1). When the key
is unset, save/load/clear are no-ops with a one-line warning — the app
keeps running in memory-only mode.

This module deliberately knows nothing about OAuth flow state or renewal.
It's pure CRUD on encrypted bytes plus a tz-aware `authenticated_at`.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db import engine
from src.etrade.crypto import decrypt, encrypt, encryption_available
from src.models import ETradeCredentials


@dataclass
class ETradeTokens:
    oauth_token: str
    oauth_token_secret: str
    authenticated_at: datetime


async def save(tokens: ETradeTokens) -> bool:
    """Encrypt and upsert tokens. Returns False if encryption isn't configured."""
    if not encryption_available():
        logger.warning(
            "TOKEN_ENCRYPTION_KEY not set; E-Trade tokens will not survive restart"
        )
        return False
    ct1 = encrypt(tokens.oauth_token)
    ct2 = encrypt(tokens.oauth_token_secret)
    if ct1 is None or ct2 is None:
        return False
    async with engine.begin() as conn:
        stmt = sqlite_insert(ETradeCredentials).values(
            id=1,
            oauth_token_ct=ct1,
            oauth_token_secret_ct=ct2,
            authenticated_at=tokens.authenticated_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "oauth_token_ct": stmt.excluded.oauth_token_ct,
                "oauth_token_secret_ct": stmt.excluded.oauth_token_secret_ct,
                "authenticated_at": stmt.excluded.authenticated_at,
            },
        )
        await conn.execute(stmt)
    return True


async def load() -> ETradeTokens | None:
    """Return the persisted tokens, or None if missing / disabled / decrypt failed."""
    if not encryption_available():
        logger.info("TOKEN_ENCRYPTION_KEY not set; skipping persisted token load")
        return None
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                select(ETradeCredentials).where(ETradeCredentials.id == 1)
            )
        ).first()
    if row is None:
        logger.debug("No persisted E-Trade credentials row")
        return None
    token = decrypt(row.oauth_token_ct)
    secret = decrypt(row.oauth_token_secret_ct)
    if token is None or secret is None:
        logger.error(
            "Failed to decrypt persisted E-Trade tokens — check TOKEN_ENCRYPTION_KEY"
        )
        return None
    # SQLite returns tz-naive datetimes regardless of column declaration;
    # we always store UTC so coerce on load.
    authenticated_at = row.authenticated_at
    if authenticated_at.tzinfo is None:
        authenticated_at = authenticated_at.replace(tzinfo=timezone.utc)
    return ETradeTokens(
        oauth_token=token,
        oauth_token_secret=secret,
        authenticated_at=authenticated_at,
    )


async def clear() -> None:
    async with engine.begin() as conn:
        await conn.execute(delete(ETradeCredentials).where(ETradeCredentials.id == 1))
