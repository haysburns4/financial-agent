"""Checks map real failure modes to ok / warn / fail without leaking values."""
import os
import threading
import time
from pathlib import Path

import httpx
import pytest
import requests
from requests_oauthlib.oauth1_session import TokenRequestDenied

from src.cli.checks import (
    Status,
    check_etrade_keys,
    check_llm_key,
    check_node,
    check_node_modules,
    check_port,
    check_shadowing,
    check_web_env,
    mask,
)
from src.cli.doctor import exit_code, run_doctor
from tests.cli_fakes import COMPLETE, FakeNetwork, FakeSystem

# ---------- LLM key ----------


def _client(status: int) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status)))


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_llm_key_accepted(provider):
    assert check_llm_key(provider, "k", _client(200)).status is Status.OK


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_llm_key_rejected(provider):
    result = check_llm_key(provider, "k", _client(401))
    assert result.status is Status.FAIL
    assert result.message == "key rejected"


def test_llm_key_network_error_is_a_warning():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    assert check_llm_key("anthropic", "k", client).status is Status.WARN


def test_llm_key_sends_the_key_where_each_provider_expects_it():
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(record))
    check_llm_key("anthropic", "a-key", client)
    check_llm_key("openai", "o-key", client)
    assert seen[0].headers["x-api-key"] == "a-key"
    assert seen[1].headers["authorization"] == "Bearer o-key"


# ---------- E-Trade keys ----------

_REJECTED = "Token request failed with code 401, response was 'oauth_problem=consumer_key_rejected'."


class _OAuth:
    """Stands in for pyetrade.ETradeOAuth; `behaviour` runs in get_request_token."""

    behaviour: Exception | None = None

    def __init__(self, consumer_key: str, consumer_secret: str) -> None:
        pass

    def get_request_token(self) -> str:
        if self.behaviour is not None:
            raise self.behaviour
        return "https://us.etrade.com/e/t/etws/authorize?key=k&token=t"


def _oauth_raising(exc: Exception | None) -> type[_OAuth]:
    return type("OAuth", (_OAuth,), {"behaviour": exc})


def test_etrade_keys_accepted():
    assert check_etrade_keys("k", "s", _oauth_raising(None)).status is Status.OK


def test_etrade_rejected_key_gets_a_readable_message():
    result = check_etrade_keys("k", "s", _oauth_raising(TokenRequestDenied(_REJECTED, response=None)))
    assert result.status is Status.FAIL
    assert "does not recognise this consumer key" in result.message


def test_etrade_unknown_problem_is_still_named():
    exc = TokenRequestDenied("oauth_problem=something_new", response=None)
    result = check_etrade_keys("k", "s", _oauth_raising(exc))
    assert result.status is Status.FAIL
    assert "something_new" in result.message


def test_etrade_network_error_is_a_warning():
    result = check_etrade_keys("k", "s", _oauth_raising(requests.ConnectionError("down")))
    assert result.status is Status.WARN


def test_etrade_timeout_is_a_warning():
    release = threading.Event()

    class Hangs(_OAuth):
        def get_request_token(self) -> str:
            release.wait(5)
            return ""

    started = time.monotonic()
    result = check_etrade_keys("k", "s", Hangs, timeout=0.05)
    release.set()
    assert result.status is Status.WARN
    assert time.monotonic() - started < 2


# ---------- shell shadowing ----------


def test_shadowing_warns_when_the_shell_differs():
    results = check_shadowing(
        {"ANTHROPIC_API_KEY": "from-dotenv", "LOG_LEVEL": "INFO"},
        {"ANTHROPIC_API_KEY": "stale-export", "LOG_LEVEL": "INFO", "HOME": "/x"},
    )
    assert [r.name for r in results] == ["ANTHROPIC_API_KEY"]
    assert "unset ANTHROPIC_API_KEY" in (results[0].fix or "")
    assert "stale-export" not in results[0].message + (results[0].fix or "")


def test_shadowing_ignores_unmanaged_and_dotenv_only_names():
    assert check_shadowing({"WATCHLIST": "AAPL"}, {"PATH": "/bin", "CUSTOM": "1"}) == []


# ---------- toolchain ----------


@pytest.mark.parametrize(("version", "status"), [("v22.1.0", Status.OK), ("v18.19.0", Status.FAIL), (None, Status.FAIL)])
def test_node_version(version, status):
    assert check_node(FakeSystem(node=version)).status is status


def test_node_modules(tmp_path):
    assert check_node_modules(tmp_path).status is Status.FAIL

    (tmp_path / "node_modules").mkdir()
    marker = tmp_path / "node_modules" / ".package-lock.json"
    marker.write_text("{}")
    lock = tmp_path / "package-lock.json"
    lock.write_text("{}")
    os.utime(marker, (1000, 1000))
    os.utime(lock, (2000, 2000))
    assert check_node_modules(tmp_path).status is Status.WARN

    os.utime(marker, (3000, 3000))
    assert check_node_modules(tmp_path).status is Status.OK


def test_busy_port_names_its_holder():
    result = check_port(8000, FakeSystem(busy_ports=[8000]))
    assert result.status is Status.WARN
    assert "python3 (pid 4242)" in result.message


def test_web_env_points_at_the_configured_api(tmp_path):
    (tmp_path / ".env.local").write_text("AGUI_URL=http://127.0.0.1:9000/agui\n")
    assert check_web_env(tmp_path, {"API_PORT": "9000"}).status is Status.OK
    assert check_web_env(tmp_path, {}).status is Status.WARN


def test_mask_shows_at_most_the_last_four():
    assert mask("sk-ant-0123456789abcdef") == "set (…cdef)"
    assert mask("short") == "set"
    assert mask(None) == "not set"


# ---------- doctor ----------


def _install(root: Path, dotenv: str) -> Path:
    (root / ".env").write_text(dotenv)
    (root / ".env").chmod(0o600)
    web = root / "web"
    (web / "node_modules").mkdir(parents=True)
    (web / ".env.local").write_text("AGUI_URL=http://127.0.0.1:8000/agui\n")
    return root


def test_doctor_passes_a_complete_install(tmp_path):
    results = run_doctor(_install(tmp_path, COMPLETE), {}, FakeSystem(), FakeNetwork())
    assert exit_code(results) == 0
    assert all(r.status is Status.OK for r in results), results


def test_doctor_fails_on_a_missing_setting(tmp_path):
    root = _install(tmp_path, COMPLETE.replace("ANTHROPIC_API_KEY", "# ANTHROPIC_API_KEY"))
    results = run_doctor(root, {}, FakeSystem(), None)
    assert exit_code(results) == 1
    assert any(r.message == "ANTHROPIC_API_KEY is not set" for r in results)


def test_doctor_fails_on_a_rejected_key(tmp_path):
    network = FakeNetwork(llm=[check_llm_key("anthropic", "k", _client(401))])
    assert exit_code(run_doctor(_install(tmp_path, COMPLETE), {}, FakeSystem(), network)) == 1


def test_doctor_warnings_do_not_fail(tmp_path):
    root = _install(tmp_path, COMPLETE)
    results = run_doctor(root, {"LOG_LEVEL": "DEBUG"}, FakeSystem(busy_ports=[3000]), None)
    assert exit_code(results) == 0
    assert {r.status for r in results} == {Status.OK, Status.WARN}


def test_doctor_uses_shell_values_like_the_app_does(tmp_path):
    root = _install(tmp_path, COMPLETE)
    results = run_doctor(root, {"API_PORT": "not-a-port"}, FakeSystem(), None)
    assert exit_code(results) == 1


def test_doctor_without_env_file_fails(tmp_path):
    assert exit_code(run_doctor(tmp_path, {}, FakeSystem(), None)) == 1


def test_doctor_requires_the_openai_package_for_openai(tmp_path):
    dotenv = COMPLETE.replace("LLM_PROVIDER=anthropic", "LLM_PROVIDER=openai") + "OPENAI_API_KEY=sk-o\n"
    results = run_doctor(_install(tmp_path, dotenv), {}, FakeSystem(modules=()), None)
    assert any(r.name == "openai package" and r.status is Status.FAIL for r in results)
