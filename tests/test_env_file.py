"""The .env editor keeps everything it does not own, and pydantic-settings reads back what it writes."""
import stat

import pytest
from dotenv import dotenv_values

from src.cli.env_file import Assignment, EnvFile, Other, UnwritableValue, parse_line, quote

SAMPLE = """\
# E-Trade
ETRADE_CONSUMER_KEY=abc123
export ETRADE_SANDBOX=true   # flip for production

SOMETHING_ELSE='kept as is'
not an assignment
LOG_LEVEL="DEBUG"
"""


def test_round_trip_is_byte_for_byte():
    assert EnvFile.parse(SAMPLE).render() == SAMPLE


def test_round_trip_keeps_a_missing_trailing_newline():
    assert EnvFile.parse("A=1\n# end").render() == "A=1\n# end"


def test_parse_reads_values_like_dotenv(tmp_path):
    path = tmp_path / ".env"
    path.write_text(SAMPLE)
    env = EnvFile.read(path)

    for key, value in dotenv_values(path).items():
        assert env.get(key) == value
    assert env.keys() == ["ETRADE_CONSUMER_KEY", "ETRADE_SANDBOX", "SOMETHING_ELSE", "LOG_LEVEL"]


def test_set_updates_in_place_and_keeps_inline_comment():
    env = EnvFile.parse(SAMPLE)
    env.set("ETRADE_SANDBOX", "false")

    lines = env.render().splitlines()
    assert lines[2] == "export ETRADE_SANDBOX=false   # flip for production"
    assert len(lines) == len(SAMPLE.splitlines())


def test_set_replaces_a_quoted_value_whole():
    env = EnvFile.parse('LOG_LEVEL="DEBUG"  # noisy\n')
    env.set("LOG_LEVEL", "INFO")
    assert env.render() == "LOG_LEVEL=INFO  # noisy\n"


def test_set_appends_a_new_key_after_everything_else():
    env = EnvFile.parse(SAMPLE)
    env.set("API_PORT", "8001")

    rendered = env.render()
    assert rendered.startswith(SAMPLE)
    assert rendered.endswith("API_PORT=8001\n")


def test_set_updates_every_duplicate():
    env = EnvFile.parse("A=1\nA=2\n")
    env.set("A", "3")
    assert env.render() == "A=3\nA=3\n"


def test_unterminated_quote_is_kept_but_not_an_assignment():
    assert parse_line('KEY="never closed') == Other('KEY="never closed')


def test_unquoted_hash_without_space_is_part_of_the_value():
    entry = parse_line("KEY=a#b")
    assert isinstance(entry, Assignment)
    assert entry.value == "a#b"


@pytest.mark.parametrize(
    "value",
    [
        "with spaces",
        "hash # inside",
        "#leading",
        "it's",
        'say "hi"',
        "back\\slash",
        "$HOME is not expanded",
        "",
        "sqlite+aiosqlite:///data/agent.db",
        "Zm9vYmFy-_base64key=",
    ],
)
def test_written_values_read_back_unchanged(tmp_path, value):
    path = tmp_path / ".env"
    env = EnvFile.parse("# header\n")
    env.set("KEY", value)
    env.write(path)

    assert dotenv_values(path)["KEY"] == value
    assert EnvFile.read(path).get("KEY") == value


def test_plain_values_are_not_quoted():
    assert quote("AAPL,MSFT") == "AAPL,MSFT"
    assert quote("https://discord.com/api/webhooks/1/x-y_z") == "https://discord.com/api/webhooks/1/x-y_z"


def test_values_that_need_it_are_quoted():
    assert quote("two words") == "'two words'"
    assert quote("a #b") == "'a #b'"


@pytest.mark.parametrize("value", ["${HOME}", "two\nlines"])
def test_values_dotenv_cannot_hold_are_rejected(value):
    with pytest.raises(UnwritableValue):
        EnvFile.parse("").set("KEY", value)


def test_written_file_is_owner_only(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=1\n")
    path.chmod(0o644)

    env = EnvFile.read(path)
    env.set("A", "2")
    env.write(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text() == "A=2\n"
    assert [p.name for p in tmp_path.iterdir()] == [".env"]


def test_missing_file_reads_as_empty(tmp_path):
    env = EnvFile.read(tmp_path / ".env")
    assert env.entries == []
    env.set("A", "1")
    assert env.render() == "A=1\n"
