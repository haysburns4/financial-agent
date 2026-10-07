"""The settings spec stays in lockstep with Settings and with the committed .env.example."""
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_core import PydanticUndefined

from src.cli.spec import BY_NAME, EXAMPLE_PATH, SPEC, Always, is_required, problem, providers_in_use, render_example

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings_cls(monkeypatch):
    # Importing src.config builds Settings(), which needs the E-Trade keys.
    monkeypatch.setenv("ETRADE_CONSUMER_KEY", "dummy")
    monkeypatch.setenv("ETRADE_CONSUMER_SECRET", "dummy")
    from src.config import Settings

    return Settings


def _as_env(value: object) -> str | None:
    """A Settings default as it would be written in .env."""
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return ",".join(value)
    return str(value)


def test_spec_covers_exactly_the_settings_fields(settings_cls):
    assert [s.name for s in SPEC] == list(settings_cls.model_fields)


def test_spec_defaults_match_settings(settings_cls):
    for spec in SPEC:
        field = settings_cls.model_fields[spec.name]
        if field.default_factory is not None:
            default = field.default_factory()
        elif field.default is PydanticUndefined:
            default = None
        else:
            default = field.default
        assert spec.default == _as_env(default), spec.name


def test_settings_without_a_default_are_always_required(settings_cls):
    for spec in SPEC:
        if settings_cls.model_fields[spec.name].is_required():
            assert spec.required_when == Always(), spec.name


def test_providers_in_use_follow_the_task_overrides():
    assert providers_in_use({}) == {"anthropic"}
    assert providers_in_use({"LLM_PROVIDER": "local"}) == {"local"}
    assert providers_in_use({"LLM_PROVIDER": "anthropic", "LLM_PROVIDER_CHAT": "local"}) == {"anthropic", "local"}
    # Both tasks overridden: LLM_PROVIDER itself serves nothing.
    assert providers_in_use(
        {"LLM_PROVIDER": "openai", "LLM_PROVIDER_CHAT": "local", "LLM_PROVIDER_SYNTHESIZER": "anthropic"}
    ) == {"local", "anthropic"}


def test_a_key_is_required_when_any_task_uses_its_provider():
    anthropic = BY_NAME["ANTHROPIC_API_KEY"]
    assert is_required(anthropic, {"LLM_PROVIDER": "local", "LLM_PROVIDER_SYNTHESIZER": "anthropic"})
    assert not is_required(anthropic, {"LLM_PROVIDER": "anthropic", "LLM_PROVIDER_CHAT": "local",
                                       "LLM_PROVIDER_SYNTHESIZER": "local"})


def test_provider_key_requirement_follows_llm_provider():
    by_name = {s.name: s for s in SPEC}
    openai, anthropic = by_name["OPENAI_API_KEY"], by_name["ANTHROPIC_API_KEY"]

    assert is_required(anthropic, {})  # anthropic is the default provider
    assert not is_required(openai, {})
    assert is_required(openai, {"LLM_PROVIDER": " OpenAI "})
    assert not is_required(anthropic, {"LLM_PROVIDER": "openai"})


def test_committed_example_matches_the_spec():
    assert EXAMPLE_PATH.read_text() == render_example(), (
        "regenerate with: uv run python -m src.cli.spec --write-example"
    )


def test_example_holds_no_secret_values():
    secrets = {s.name for s in SPEC if s.secret}
    for line in render_example().splitlines():
        key, _, value = line.partition("=")
        if key in secrets:
            assert value == "", key


def test_cli_package_does_not_import_config():
    # A fresh interpreter with no E-Trade keys: importing src.config there would raise.
    code = (
        "import sys\n"
        "import src.cli.__main__, src.cli.checks, src.cli.doctor, src.cli.env_file\n"
        "import src.cli.launcher, src.cli.login, src.cli.spec, src.cli.supervisor\n"
        "import src.cli.ui, src.cli.wizard\n"
        "assert 'src.config' not in sys.modules, 'src.cli imported src.config'\n"
    )
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT / "tests", env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("ETRADE_CONSUMER_KEY", "", "ETRADE_CONSUMER_KEY is not set"),
        ("ETRADE_CONSUMER_KEY", "has space", "ETRADE_CONSUMER_KEY must be a single word with no spaces"),
        ("ETRADE_SANDBOX", "maybe", "ETRADE_SANDBOX must be true or false"),
        ("ETRADE_SANDBOX", "False", None),
        ("PRICE_POLL_MINUTES", "0", "PRICE_POLL_MINUTES must be between 1 and 1440"),
        ("PRICE_POLL_MINUTES", "", "PRICE_POLL_MINUTES must be a whole number"),
        ("WATCHLIST", "AAPL,BRK.B", None),
        ("WATCHLIST", " , ", "WATCHLIST must list at least one ticker"),
        ("LLM_PROVIDER", "gemini", "LLM_PROVIDER must be one of: anthropic, openai, local"),
        ("DISCORD_WEBHOOK_URL", "", None),
        ("DISCORD_WEBHOOK_URL", "http://x", "DISCORD_WEBHOOK_URL must be an https:// URL"),
        ("TOKEN_ENCRYPTION_KEY", "not-a-key", "TOKEN_ENCRYPTION_KEY is not a valid Fernet key"),
    ],
)
def test_problem(name, value, expected):
    assert problem(BY_NAME[name], {name: value}) == expected


def test_absent_optional_setting_uses_its_default():
    assert problem(BY_NAME["LOG_LEVEL"], {}) is None
