"""Every setting in src/config.py's Settings, described for setup tooling.

This mirrors Settings by hand rather than importing it (see src/cli/__init__.py);
tests/test_spec.py fails if the two drift apart.

    uv run python -m src.cli.spec                  # print .env.example
    uv run python -m src.cli.spec --write-example  # regenerate it on disk
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.cli.env_file import quote, write_atomic

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / ".env.example"


@dataclass(frozen=True)
class Always:
    pass


@dataclass(frozen=True)
class Never:
    pass


@dataclass(frozen=True)
class WhenEquals:
    """Required when another setting has this value (case-insensitive)."""

    key: str
    value: str


RequiredWhen = Always | Never | WhenEquals


@dataclass(frozen=True)
class SettingSpec:
    name: str
    # As written in .env; None when there is no default.
    default: str | None
    secret: bool
    help: str
    required_when: RequiredWhen = Never()


SPEC: tuple[SettingSpec, ...] = (
    SettingSpec(
        "ETRADE_CONSUMER_KEY", None, True,
        "E-Trade API consumer key (developer.etrade.com).", Always(),
    ),
    SettingSpec(
        "ETRADE_CONSUMER_SECRET", None, True,
        "E-Trade API consumer secret, paired with the key above.", Always(),
    ),
    SettingSpec(
        "ETRADE_SANDBOX", "true", False,
        "true for E-Trade's sandbox; false for your real account (needs production keys).",
    ),
    SettingSpec("LLM_PROVIDER", "anthropic", False, "LLM provider: anthropic | openai."),
    SettingSpec("LLM_CHAT_MODEL", "claude-opus-5", False, "Model for the chat agent."),
    SettingSpec(
        "LLM_SYNTHESIS_MODEL", "claude-sonnet-5", False,
        "Model for scheduled signal synthesis.",
    ),
    SettingSpec(
        "ANTHROPIC_API_KEY", None, True,
        "Anthropic API key.", WhenEquals("LLM_PROVIDER", "anthropic"),
    ),
    SettingSpec(
        "OPENAI_API_KEY", None, True,
        "OpenAI API key (also: uv sync --extra openai).", WhenEquals("LLM_PROVIDER", "openai"),
    ),
    SettingSpec(
        "DATABASE_URL", "sqlite+aiosqlite:///data/agent.db", False,
        "SQLAlchemy async database URL.",
    ),
    SettingSpec(
        "WATCHLIST", "AAPL,MSFT,GOOGL,AMZN,NVDA", False,
        "Comma-separated tickers to track.",
    ),
    SettingSpec("PRICE_POLL_MINUTES", "5", False, "Minutes between price refreshes."),
    SettingSpec("PORTFOLIO_POLL_MINUTES", "15", False, "Minutes between portfolio refreshes."),
    SettingSpec(
        "DISCORD_WEBHOOK_URL", None, True,
        "Discord webhook for alerts; leave blank to disable.",
    ),
    SettingSpec(
        "TOKEN_ENCRYPTION_KEY", None, True,
        "Fernet key that keeps the E-Trade login across restarts; blank keeps tokens in memory only.",
    ),
    SettingSpec("LOG_LEVEL", "INFO", False, "Log level: DEBUG | INFO | WARNING | ERROR."),
    SettingSpec("API_HOST", "127.0.0.1", False, "Interface the API binds to; keep it on loopback."),
    SettingSpec("API_PORT", "8000", False, "Port the API listens on."),
)


def is_required(spec: SettingSpec, values: Mapping[str, str]) -> bool:
    """Whether `spec` must be set, given the other values (unset ones take their default)."""
    match spec.required_when:
        case Always():
            return True
        case Never():
            return False
        case WhenEquals(key=key, value=value):
            other = values.get(key, _default(key))
            return other.strip().lower() == value.lower()


def _default(name: str) -> str:
    for spec in SPEC:
        if spec.name == name:
            return spec.default or ""
    raise KeyError(name)


def _requirement(rule: RequiredWhen) -> str | None:
    match rule:
        case Always():
            return "Required."
        case Never():
            return None
        case WhenEquals(key=key, value=value):
            return f"Required when {key}={value}."


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
