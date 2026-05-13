"""Symmetric encryption for persisted E-Trade tokens.

Uses Fernet (AES-128-CBC + HMAC) with a key sourced from
`settings.TOKEN_ENCRYPTION_KEY`. If the key is unset, persistence is disabled
and the helpers return None — callers should fall back to in-memory-only.
Generate a key with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken
from loguru import logger

from src.config import settings


@lru_cache(maxsize=1)
def _cipher() -> Fernet | None:
    key = settings.TOKEN_ENCRYPTION_KEY
    if not key:
        return None
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError) as exc:
        logger.error("TOKEN_ENCRYPTION_KEY is not a valid Fernet key: {}", exc)
        return None


def encryption_available() -> bool:
    return _cipher() is not None


def encrypt(plaintext: str) -> bytes | None:
    c = _cipher()
    if c is None:
        return None
    return c.encrypt(plaintext.encode("utf-8"))


def decrypt(ciphertext: bytes) -> str | None:
    c = _cipher()
    if c is None:
        return None
    try:
        return c.decrypt(ciphertext).decode("utf-8")
    except InvalidToken:
        return None
