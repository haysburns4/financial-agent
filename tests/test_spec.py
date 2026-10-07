"""The settings spec stays in lockstep with Settings and with the committed .env.example."""
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_core import PydanticUndefined

from src.cli.spec import EXAMPLE_PATH, SPEC, Always, WhenEquals, is_required, render_example

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


def test_conditional_rules_name_real_settings():
    names = {s.name for s in SPEC}
    for spec in SPEC:
        if isinstance(spec.required_when, WhenEquals):
            assert spec.required_when.key in names


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
        "import sys, src.cli.spec, src.cli.env_file\n"
        "assert 'src.config' not in sys.modules, 'src.cli imported src.config'\n"
    )
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT / "tests", env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
