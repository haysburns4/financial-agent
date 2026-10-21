from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # LLM_PROVIDER=local: any OpenAI-compatible server.
    LOCAL_LLM_SERVER: str = "mlx"  # mlx | ollama | llamacpp | lmstudio | vllm | openai_compat
    LOCAL_LLM_BASE_URL: str = "http://localhost:8080/v1"
    LOCAL_LLM_MODEL: str = "mlx-community/Qwen3-14B-4bit"
    # Per-task models on the same server; unset means LOCAL_LLM_MODEL.
    LOCAL_LLM_MODEL_CHAT: str | None = None
    LOCAL_LLM_MODEL_SYNTHESIZER: str | None = None
    # Unset matches the family from the model id; set it for a model the
    # patterns do not recognise.
    LOCAL_LLM_MODEL_FAMILY: str | None = None
    LOCAL_LLM_TIMEOUT_SECONDS: int = 180
    LLM_CONTEXT_BUDGET: int = 120_000
    LOCAL_LLM_CONTEXT_BUDGET: int = 24_000

    # Per-task provider overrides; unset means LLM_PROVIDER. Resolved in one
    # place, src/llm/factory.py backend_for(task).
    LLM_PROVIDER_CHAT: str | None = None
    LLM_PROVIDER_SYNTHESIZER: str | None = None

    DATABASE_URL: str = "sqlite+aiosqlite:///data/agent.db"

    WATCHLIST_MAX_SIZE: int = 50

    PRICE_POLL_MINUTES: int = 5
    PORTFOLIO_POLL_MINUTES: int = 15
    SNAPSHOT_ON_STARTUP: bool = True


    # Fernet key for encrypting persisted E-Trade tokens. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # If unset, tokens are kept in memory only (current behavior, no persistence).
    TOKEN_ENCRYPTION_KEY: str | None = None

    LOG_LEVEL: str = "INFO"

    # Loopback by default: the API has no auth of its own.
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000


settings = Settings()
