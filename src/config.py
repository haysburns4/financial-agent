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

    # Only the keys of providers in use are required; build_backend() raises
    # LLMConfigError at startup if one is missing.
    LLM_PROVIDER: str = "anthropic"  # anthropic | openai | local
    LLM_CHAT_MODEL: str = "claude-opus-5"
    LLM_SYNTHESIS_MODEL: str = "claude-sonnet-5"
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    # LLM_PROVIDER=local: an OpenAI-compatible MLX server (mlx_lm.server).
    LOCAL_LLM_BASE_URL: str = "http://localhost:8080/v1"
    LOCAL_LLM_MODEL: str = "mlx-community/Qwen3-14B-4bit"
    LOCAL_LLM_TIMEOUT_SECONDS: int = 180

    # Per-task provider overrides; unset means LLM_PROVIDER. Resolved in one
    # place, src/llm/factory.py backend_for(task).
    LLM_PROVIDER_CHAT: str | None = None
    LLM_PROVIDER_SYNTHESIZER: str | None = None

    DATABASE_URL: str = "sqlite+aiosqlite:///data/agent.db"

    WATCHLIST: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    )

    PRICE_POLL_MINUTES: int = 5
    PORTFOLIO_POLL_MINUTES: int = 15
    SNAPSHOT_ON_STARTUP: bool = True

    DISCORD_WEBHOOK_URL: str | None = None

    # Fernet key for encrypting persisted E-Trade tokens. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # If unset, tokens are kept in memory only (current behavior, no persistence).
    TOKEN_ENCRYPTION_KEY: str | None = None

    LOG_LEVEL: str = "INFO"

    # Loopback by default: the API has no auth of its own.
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    @field_validator("WATCHLIST", mode="before")
    @classmethod
    def _split_watchlist(cls, v):
        if isinstance(v, str):
            return [t.strip().upper() for t in v.split(",") if t.strip()]
        return v


settings = Settings()
