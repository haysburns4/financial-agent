from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    ETRADE_CONSUMER_KEY: str
    ETRADE_CONSUMER_SECRET: str
    ETRADE_SANDBOX: bool = True

    DATABASE_URL: str = "sqlite+aiosqlite:///data/agent.db"

    WATCHLIST: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    )

    PRICE_POLL_MINUTES: int = 5
    PORTFOLIO_POLL_MINUTES: int = 15

    DISCORD_WEBHOOK_URL: str | None = None

    # Fernet key for encrypting persisted E-Trade tokens. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # If unset, tokens are kept in memory only (current behavior, no persistence).
    TOKEN_ENCRYPTION_KEY: str | None = None

    LOG_LEVEL: str = "INFO"

    @field_validator("WATCHLIST", mode="before")
    @classmethod
    def _split_watchlist(cls, v):
        if isinstance(v, str):
            return [t.strip().upper() for t in v.split(",") if t.strip()]
        return v


settings = Settings()
