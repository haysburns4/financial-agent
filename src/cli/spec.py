"""Every setting in src/config.py's Settings, described for setup tooling.

This mirrors Settings by hand rather than importing it (see src/cli/__init__.py);
tests/test_spec.py fails if the two drift apart.

    uv run python -m src.cli.spec                  # print .env.example
    uv run python -m src.cli.spec --write-example  # regenerate it on disk
"""
from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet

from src.cli.env_file import quote, write_atomic

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / ".env.example"


@dataclass(frozen=True)
class Always:
    pass


@dataclass(frozen=True)
class Never:
    pass


@dataclass(frozen=True)
class WhenProviderUsed:
    """Required when any LLM task runs on this provider: LLM_PROVIDER, or a
    per-task override (LLM_PROVIDER_CHAT / LLM_PROVIDER_SYNTHESIZER)."""

    provider: str


RequiredWhen = Always | Never | WhenProviderUsed


@dataclass(frozen=True)
class Credential:
    """An opaque key or secret: anything without whitespace."""


@dataclass(frozen=True)
class Text:
    pass


@dataclass(frozen=True)
class Boolean:
    pass


@dataclass(frozen=True)
class Integer:
    minimum: int
    maximum: int


@dataclass(frozen=True)
class Choice:
    options: tuple[str, ...]


@dataclass(frozen=True)
class Tickers:
    pass


@dataclass(frozen=True)
class HttpsUrl:
    pass


@dataclass(frozen=True)
class FernetKey:
    pass


ValueRule = Credential | Text | Boolean | Integer | Choice | Tickers | HttpsUrl | FernetKey


@dataclass(frozen=True)
class SettingSpec:
    name: str
    # As written in .env; None when there is no default.
    default: str | None
    secret: bool
    help: str
    rule: ValueRule
    required_when: RequiredWhen = Never()


PROVIDERS = ("anthropic", "openai", "local")
# (chat model, synthesis model) offered when a provider is picked.
PROVIDER_MODELS: Mapping[str, tuple[str, str]] = {
    "anthropic": ("claude-opus-5", "claude-sonnet-5"),
    "openai": ("gpt-5", "gpt-5-mini"),
}
# The tasks' provider overrides; unset means LLM_PROVIDER. Mirrors
# resolve_task() in src/llm/factory.py, which src/cli cannot import.
TASK_PROVIDER_SETTINGS = ("LLM_PROVIDER_CHAT", "LLM_PROVIDER_SYNTHESIZER")
# Providers that need the openai package: OpenAI itself, and the local MLX
# server, which speaks its API.
OPENAI_SDK_PROVIDERS = ("openai", "local")
PROVIDER_KEYS: Mapping[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
# Pydantic's accepted spellings, lower-cased.
_TRUE = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE = frozenset({"0", "false", "f", "no", "n", "off"})
_TICKER = re.compile(r"^[A-Za-z0-9.^=-]{1,15}$")

SPEC: tuple[SettingSpec, ...] = (
    SettingSpec(
        "ETRADE_CONSUMER_KEY", None, True,
        "E-Trade API consumer key (developer.etrade.com).", Credential(), Always(),
    ),
    SettingSpec(
        "ETRADE_CONSUMER_SECRET", None, True,
        "E-Trade API consumer secret, paired with the key above.", Credential(), Always(),
    ),
    SettingSpec(
        "ETRADE_SANDBOX", "true", False,
        "true for E-Trade's sandbox; false for your real account (needs production keys).",
        Boolean(),
    ),
    SettingSpec(
        "LLM_PROVIDER", "anthropic", False, "LLM provider: anthropic | openai | local (an MLX server).", Choice(PROVIDERS)
    ),
    SettingSpec("LLM_CHAT_MODEL", "claude-opus-5", False, "Model for the chat agent.", Text()),
    SettingSpec(
        "LLM_SYNTHESIS_MODEL", "claude-sonnet-5", False,
        "Model for scheduled signal synthesis.", Text(),
    ),
    SettingSpec(
        "ANTHROPIC_API_KEY", None, True,
        "Anthropic API key.", Credential(), WhenProviderUsed("anthropic"),
    ),
    SettingSpec(
        "OPENAI_API_KEY", None, True,
        "OpenAI API key (also: uv sync --extra openai).", Credential(), WhenProviderUsed("openai"),
    ),
    SettingSpec(
        "LOCAL_LLM_BASE_URL", "http://localhost:8080/v1", False,
        "Provider local: the MLX server's OpenAI-compatible endpoint (mlx_lm.server).", Text(),
    ),
    SettingSpec(
        "LOCAL_LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit", False,
        "Provider local: the model the MLX server serves, used by every task routed to it.", Text(),
    ),
    SettingSpec(
        "LOCAL_LLM_TIMEOUT_SECONDS", "180", False,
        "Provider local: request timeout; a large model's first token can take tens of seconds.",
        Integer(1, 3600),
    ),
    SettingSpec(
        "LLM_PROVIDER_CHAT", None, False,
        "Provider for the chat agent; blank uses LLM_PROVIDER.", Choice(PROVIDERS),
    ),
    SettingSpec(
        "LLM_PROVIDER_SYNTHESIZER", None, False,
        "Provider for the signal synthesizer; blank uses LLM_PROVIDER.", Choice(PROVIDERS),
    ),
    SettingSpec(
        "DATABASE_URL", "sqlite+aiosqlite:///data/agent.db", False,
        "SQLAlchemy async database URL.", Text(),
    ),
    SettingSpec(
        "WATCHLIST", "AAPL,MSFT,GOOGL,AMZN,NVDA", False,
        "Comma-separated tickers to track.", Tickers(),
    ),
    SettingSpec(
        "PRICE_POLL_MINUTES", "5", False, "Minutes between price refreshes.", Integer(1, 1440)
    ),
    SettingSpec(
        "PORTFOLIO_POLL_MINUTES", "15", False,
        "Minutes between portfolio refreshes.", Integer(1, 1440),
    ),
    SettingSpec(
        "DISCORD_WEBHOOK_URL", None, True,
        "Discord webhook for alerts; leave blank to disable.", HttpsUrl(),
    ),
    SettingSpec(
        "TOKEN_ENCRYPTION_KEY", None, True,
        "Fernet key that keeps the E-Trade login across restarts; blank keeps tokens in memory only.",
        FernetKey(),
    ),
    SettingSpec(
        "LOG_LEVEL", "INFO", False, "Log level: DEBUG | INFO | WARNING | ERROR.",
        Choice(("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")),
    ),
    SettingSpec(
        "API_HOST", "127.0.0.1", False, "Interface the API binds to; keep it on loopback.", Text()
    ),
    SettingSpec("API_PORT", "8000", False, "Port the API listens on.", Integer(1, 65535)),
)

BY_NAME: Mapping[str, SettingSpec] = {spec.name: spec for spec in SPEC}


def is_required(spec: SettingSpec, values: Mapping[str, str]) -> bool:
    """Whether `spec` must be set, given the other values (unset ones take their default)."""
    match spec.required_when:
        case Always():
            return True
        case Never():
            return False
        case WhenProviderUsed(provider=provider):
            return provider in providers_in_use(values)


def providers_in_use(values: Mapping[str, str]) -> set[str]:
    """Every provider some LLM task runs on: each task's override, else LLM_PROVIDER."""
    base = (values.get("LLM_PROVIDER") or BY_NAME["LLM_PROVIDER"].default or "").strip().lower()
    return {(values.get(name) or base).strip().lower() for name in TASK_PROVIDER_SETTINGS}


def problem(spec: SettingSpec, values: Mapping[str, str]) -> str | None:
    """Why `spec`'s value in `values` would not work, or None if it is fine.

    Absent means "use the default". An empty value is unset only for settings
    whose default is None; for the rest Settings would try to parse "".
    """
    value = values.get(spec.name)
    if not value and is_required(spec, values):
        return f"{spec.name} is not set"
    if value is None or (value == "" and spec.default is None):
        return None
    reason = _rule_problem(spec.rule, value)
    return f"{spec.name} {reason}" if reason else None


def _rule_problem(rule: ValueRule, value: str) -> str | None:
    match rule:
        case Credential():
            if not value or any(ch.isspace() for ch in value):
                return "must be a single word with no spaces"
        case Text():
            if not value.strip():
                return "must not be blank"
        case Boolean():
            if value.strip().lower() not in _TRUE | _FALSE:
                return "must be true or false"
        case Integer(minimum=lo, maximum=hi):
            try:
                number = int(value)
            except ValueError:
                return "must be a whole number"
            if not lo <= number <= hi:
                return f"must be between {lo} and {hi}"
        case Choice(options=options):
            if value.strip().lower() not in {o.lower() for o in options}:
                return f"must be one of: {', '.join(options)}"
        case Tickers():
            tickers = [t.strip() for t in value.split(",") if t.strip()]
            if not tickers:
                return "must list at least one ticker"
            bad = [t for t in tickers if not _TICKER.match(t)]
            if bad:
                return f"has invalid tickers: {', '.join(bad)}"
        case HttpsUrl():
            if not value.startswith("https://"):
                return "must be an https:// URL"
        case FernetKey():
            try:
                Fernet(value.encode())
            except ValueError:
                return "is not a valid Fernet key"
    return None


def parse_bool(value: str) -> bool:
    return value.strip().lower() in _TRUE


def _requirement(rule: RequiredWhen) -> str | None:
    match rule:
        case Always():
            return "Required."
        case Never():
            return None
        case WhenProviderUsed(provider=provider):
            return f"Required when LLM_PROVIDER (or a task's override) is {provider}."


def render_example() -> str:
    lines = [
        "# Copy to .env and fill in, or let the setup wizard write .env for you.",
        "# Variables exported in your shell override .env.",
    ]
    for spec in SPEC:
        lines.append("")
        lines.append(f"# {spec.help}")
        requirement = _requirement(spec.required_when)
        if requirement:
            lines.append(f"# {requirement}")
        value = "" if spec.secret or spec.default is None else quote(spec.default)
        lines.append(f"{spec.name}={value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-example", action="store_true", help=f"write {EXAMPLE_PATH.name}")
    args = parser.parse_args()
    if args.write_example:
        # Holds no secrets and is committed, so world-readable is fine.
        write_atomic(EXAMPLE_PATH, render_example(), mode=0o644)
        print(f"wrote {EXAMPLE_PATH}")
    else:
        print(render_example(), end="")


if __name__ == "__main__":
    main()
