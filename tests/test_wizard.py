"""The setup wizard asks only for what is missing, re-asks rejected keys, and keeps the rest of .env."""
import shutil
from pathlib import Path

import pytest

from src.cli.checks import fail
from src.cli.env_file import EnvFile
from src.cli.wizard import OPENAI_SYNC, Wizard
from tests.cli_fakes import (
    ANTHROPIC_KEY,
    COMPLETE,
    ETRADE_KEY,
    ETRADE_SECRET,
    TOKEN_KEY,
    FakeNetwork,
    FakeSystem,
    ScriptedPrompter,
)

REPO = Path(__file__).resolve().parents[1]

@pytest.fixture
def root(tmp_path):
    shutil.copy(REPO / ".env.example", tmp_path / ".env.example")
    (tmp_path / "web").mkdir()
    shutil.copy(REPO / "web" / ".env.local.example", tmp_path / "web" / ".env.local.example")
    return tmp_path


def run(
    root: Path,
    prompter: ScriptedPrompter,
    network: FakeNetwork | None = None,
    system: FakeSystem | None = None,
    ask_all: bool = False,
) -> None:
    Wizard(root, prompter, system or FakeSystem(), network or FakeNetwork(), {}, ask_all).run()
    assert not prompter.script, f"never asked: {list(prompter.script)}"


def env(root: Path) -> EnvFile:
    return EnvFile.read(root / ".env")


def test_first_run_asks_the_whole_flow_in_order(root):
    io = ScriptedPrompter(
        [
            ("environment", "Sandbox"),
            ("consumer key", ETRADE_KEY),
            ("consumer secret", ETRADE_SECRET),
            ("provider", "anthropic"),
            ("anthropic API key", ANTHROPIC_KEY),
            ("Chat model", "<default>"),
            ("Synthesis model", "<default>"),
            ("Watchlist", "aapl, nvda"),
        ]
    )
    run(root, io)

    written = env(root)
    assert written.get("ETRADE_CONSUMER_KEY") == ETRADE_KEY
    assert written.get("ANTHROPIC_API_KEY") == ANTHROPIC_KEY
    assert written.get("LLM_CHAT_MODEL") == "claude-opus-5"
    assert written.get("WATCHLIST") == "AAPL,NVDA"
    assert written.get("TOKEN_ENCRYPTION_KEY")
    assert (root / "web" / ".env.local").exists()
    # The template's comments came along.
    assert "# E-Trade API consumer key" in (root / ".env").read_text()


def test_asks_only_for_missing_values(root):
    (root / ".env").write_text(COMPLETE.replace(f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}\n", ""))
    io = ScriptedPrompter([("anthropic API key", ANTHROPIC_KEY)])
    network = FakeNetwork()

    run(root, io, network)

    assert io.asked == ["anthropic API key"]
    assert env(root).get("ANTHROPIC_API_KEY") == ANTHROPIC_KEY
    # Credentials already in .env were still checked.
    assert network.etrade_calls == [(ETRADE_KEY, ETRADE_SECRET)]


def test_complete_env_asks_nothing(root):
    (root / ".env").write_text(COMPLETE)
    io = ScriptedPrompter()
    run(root, io)
    assert io.asked == []


def test_invalid_value_is_asked_for(root):
    (root / ".env").write_text(COMPLETE.replace("WATCHLIST=AAPL,MSFT", "WATCHLIST=AAPL,NOT A TICKER"))
    io = ScriptedPrompter([("Watchlist", "AAPL,MSFT,NVDA")])
    run(root, io)
    assert env(root).get("WATCHLIST") == "AAPL,MSFT,NVDA"


def test_rejected_etrade_keys_are_asked_again(root):
    (root / ".env").write_text(COMPLETE.replace(f"ETRADE_CONSUMER_KEY={ETRADE_KEY}\n", ""))
    rejected = fail("E-Trade keys", "E-Trade does not recognise this consumer key", "copy it again")
    network = FakeNetwork(etrade=[rejected])
    io = ScriptedPrompter(
        [
            ("consumer key", "wrong-key"),
            ("consumer secret", ""),  # Enter keeps the secret already in .env
            ("Enter them again", "y"),
            ("consumer key", ETRADE_KEY),
            ("consumer secret", ""),
        ]
    )

    run(root, io, network)

    assert network.etrade_calls == [("wrong-key", ETRADE_SECRET), (ETRADE_KEY, ETRADE_SECRET)]
    assert env(root).get("ETRADE_CONSUMER_KEY") == ETRADE_KEY
    assert any("does not recognise" in line for line in io.output)


def test_a_stored_key_the_provider_rejects_is_asked_again(root):
    (root / ".env").write_text(COMPLETE)
    network = FakeNetwork(llm=[fail("anthropic API key", "key rejected")])
    io = ScriptedPrompter([("anthropic API key", "sk-ant-new-0000000000000000wxyz")])

    run(root, io, network)

    assert env(root).get("ANTHROPIC_API_KEY") == "sk-ant-new-0000000000000000wxyz"


def test_unknown_lines_survive(root):
    (root / ".env").write_text("# my notes\nCUSTOM_THING='keep me'\n\n" + COMPLETE.replace("WATCHLIST=AAPL,MSFT\n", ""))
    io = ScriptedPrompter([("Watchlist", "<default>")])

    run(root, io)

    text = (root / ".env").read_text()
    assert text.startswith("# my notes\nCUSTOM_THING='keep me'\n\n")
    assert env(root).get("WATCHLIST") == "AAPL,MSFT,GOOGL,AMZN,NVDA"


def test_token_key_is_generated_once(root):
    (root / ".env").write_text(COMPLETE.replace(f"TOKEN_ENCRYPTION_KEY={TOKEN_KEY}\n", ""))
    run(root, ScriptedPrompter())
    first = env(root).get("TOKEN_ENCRYPTION_KEY")

    run(root, ScriptedPrompter())

    assert first
    assert env(root).get("TOKEN_ENCRYPTION_KEY") == first


def test_existing_token_key_is_kept(root):
    (root / ".env").write_text(COMPLETE)
    run(root, ScriptedPrompter())
    assert env(root).get("TOKEN_ENCRYPTION_KEY") == TOKEN_KEY


def test_no_secret_is_ever_shown(root):
    (root / ".env").write_text(COMPLETE)
    io = ScriptedPrompter(
        [
            ("environment", "Sandbox"),
            ("consumer key", ""),
            ("consumer secret", ""),
            ("provider", "anthropic"),
            ("anthropic API key", ""),
            ("Chat model", "<default>"),
            ("Synthesis model", "<default>"),
            ("Watchlist", "<default>"),
        ]
    )
    run(root, io, ask_all=True)

    shown = "\n".join(io.asked + io.output)
    for secret in (ETRADE_KEY, ETRADE_SECRET, ANTHROPIC_KEY, TOKEN_KEY):
        assert secret not in shown
    assert "set (…abcd)" in shown


def test_switching_to_openai_suggests_its_models_and_installs_the_package(root):
    (root / ".env").write_text(COMPLETE)
    system = FakeSystem(modules=())
    io = ScriptedPrompter(
        [
            ("environment", "Sandbox"),
            ("consumer key", ""),
            ("consumer secret", ""),
            ("provider", "openai"),
            ("openai package is not installed", "y"),
            ("openai API key", "sk-openai-000000000000000000ef"),
            ("Chat model", "<default>"),
            ("Synthesis model", "<default>"),
            ("Watchlist", "<default>"),
        ]
    )

    run(root, io, system=system, ask_all=True)

    assert system.commands == [OPENAI_SYNC]
    assert env(root).get("LLM_CHAT_MODEL") == "gpt-5"
    assert env(root).get("LLM_SYNTHESIS_MODEL") == "gpt-5-mini"


def test_switching_to_the_real_account_asks_for_new_keys(root):
    (root / ".env").write_text(COMPLETE.replace("ETRADE_SANDBOX=true\n", ""))
    io = ScriptedPrompter(
        [
            ("environment", "Real account"),
            ("consumer key", "prod-key-0000000000000000"),
            ("consumer secret", "prod-secret-00000000000000"),
        ]
    )
    run(root, io)

    assert env(root).get("ETRADE_SANDBOX") == "false"
    assert env(root).get("ETRADE_CONSUMER_KEY") == "prod-key-0000000000000000"


def test_agui_url_is_offered_a_fix(root):
    (root / ".env").write_text(COMPLETE)
    (root / "web" / ".env.local").write_text("AGUI_URL=http://localhost:8000/agui\n")
    io = ScriptedPrompter([("AGUI_URL", "y")])

    run(root, io)

    assert EnvFile.read(root / "web" / ".env.local").get("AGUI_URL") == "http://127.0.0.1:8000/agui"


def test_offline_skips_key_checks(root):
    (root / ".env").write_text(COMPLETE)
    io = ScriptedPrompter()
    Wizard(root, io, FakeSystem(), None, {}).run()
    assert not io.script
    assert io.asked == []
