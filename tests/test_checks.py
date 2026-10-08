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
    check_local_llm,
    check_node,
    check_node_modules,
    check_port,
    check_shadowing,
    check_web_env,
    hf_cache_dir,
    local_server_target,
    mask,
    model_downloaded,
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
    assert check_node_modules(tmp_path).status is Status.WARN

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


def test_doctor_without_env_file_fails_when_nothing_is_exported(tmp_path):
    results = run_doctor(tmp_path, {}, FakeSystem(), None)
    assert exit_code(results) == 1
    assert any(r.message == "ETRADE_CONSUMER_KEY is not set" for r in results)


def test_doctor_accepts_settings_from_the_environment_alone(tmp_path):
    environ = {"ETRADE_CONSUMER_KEY": "k", "ETRADE_CONSUMER_SECRET": "s", "ANTHROPIC_API_KEY": "a"}
    results = run_doctor(tmp_path, environ, FakeSystem(), None)
    assert exit_code(results) == 0
    assert any(r.name == ".env" and r.status is Status.WARN for r in results)


def test_doctor_requires_the_openai_package_for_openai(tmp_path):
    dotenv = COMPLETE.replace("LLM_PROVIDER=anthropic", "LLM_PROVIDER=openai") + "OPENAI_API_KEY=sk-o\n"
    results = run_doctor(_install(tmp_path, dotenv), {}, FakeSystem(modules=()), None)
    assert any(r.name == "openai package" and r.status is Status.FAIL for r in results)


def test_doctor_checks_the_local_server_instead_of_a_key(tmp_path):
    dotenv = COMPLETE.replace("LLM_PROVIDER=anthropic", "LLM_PROVIDER=local")
    network = FakeNetwork()
    results = run_doctor(_install(tmp_path, dotenv), {}, FakeSystem(), network)
    assert exit_code(results) == 0
    assert network.local_calls == ["http://localhost:8080/v1"] and network.llm_calls == []
    assert any(r.name == "openai package" for r in results)  # the local provider needs the SDK too


def test_doctor_checks_every_provider_a_task_uses(tmp_path):
    dotenv = COMPLETE + "LLM_PROVIDER_SYNTHESIZER=local\n"
    network = FakeNetwork()
    run_doctor(_install(tmp_path, dotenv), {}, FakeSystem(), network)
    assert network.llm_calls and network.llm_calls[0][0] == "anthropic"
    assert network.local_calls == ["http://localhost:8080/v1"]


# ---------- local LLM (mlx_lm.server) ----------

LOCAL_MODEL = "mlx-community/Qwen3-14B-4bit"


def _cache_model(cache: Path, model: str = LOCAL_MODEL, weights: bool = True) -> None:
    snapshot = cache / f"models--{model.replace('/', '--')}" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    if weights:
        (snapshot / "model.safetensors").write_text("x")


def _local_values(**extra: str) -> dict[str, str]:
    return {"LLM_PROVIDER": "local", **extra}


def _by_name(results):
    return {r.name: r for r in results}


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"HF_HUB_CACHE": "/c/hub"}, Path("/c/hub")),
        ({"HF_HOME": "/h"}, Path("/h/hub")),
        ({"XDG_CACHE_HOME": "/x"}, Path("/x/huggingface/hub")),
        ({}, Path.home() / ".cache" / "huggingface" / "hub"),
    ],
)
def test_hf_cache_dir_follows_huggingface_precedence(environ, expected):
    assert hf_cache_dir(environ) == expected


def test_model_downloaded_needs_weights_not_just_a_snapshot(tmp_path):
    assert not model_downloaded(LOCAL_MODEL, tmp_path)
    _cache_model(tmp_path, weights=False)
    assert not model_downloaded(LOCAL_MODEL, tmp_path)  # an interrupted download
    (next((tmp_path).rglob("abc123")) / "model.safetensors").write_text("x")
    assert model_downloaded(LOCAL_MODEL, tmp_path)


def test_model_given_as_a_local_directory_counts_as_downloaded(tmp_path):
    assert model_downloaded(str(tmp_path), tmp_path / "empty-cache")


@pytest.mark.parametrize(
    ("url", "host", "port"),
    [
        ("http://localhost:8080/v1", "localhost", 8080),
        ("http://127.0.0.1:9000/v1", "127.0.0.1", 9000),
        ("http://gpu-box/v1", "gpu-box", 80),
    ],
)
def test_local_server_target_parses_the_base_url(url, host, port):
    target = local_server_target({"LOCAL_LLM_BASE_URL": url})
    assert target is not None and (target.host, target.port) == (host, port)
    assert target.model == LOCAL_MODEL


@pytest.mark.parametrize("url", ["localhost:8080", "http://:8080/v1", "http://localhost:99999/v1"])
def test_local_server_target_rejects_unusable_urls(url):
    assert local_server_target({"LOCAL_LLM_BASE_URL": url}) is None


def test_local_llm_checks_only_run_when_a_task_uses_local(tmp_path):
    assert check_local_llm({"LLM_PROVIDER": "anthropic"}, {"HF_HUB_CACHE": str(tmp_path)}, FakeSystem()) == []


def test_local_llm_checks_when_everything_is_ready(tmp_path):
    _cache_model(tmp_path)
    system = FakeSystem(modules=("openai", "mlx_lm"), busy_ports=(8080,))

    results = _by_name(check_local_llm(_local_values(), {"HF_HUB_CACHE": str(tmp_path)}, system))

    assert {name: r.status for name, r in results.items()} == {
        "mlx-lm": Status.OK, "local model": Status.OK, "local server port": Status.OK,
    }


def test_local_llm_checks_warn_about_a_download_and_a_stopped_server(tmp_path):
    results = _by_name(check_local_llm(_local_values(), {"HF_HUB_CACHE": str(tmp_path)}, FakeSystem()))

    assert results["mlx-lm"].status is Status.WARN
    assert results["local model"].status is Status.WARN and "several GB" in results["local model"].message
    assert results["local server port"].status is Status.WARN
    assert "--with-local-llm" in (results["local server port"].fix or "")


def test_local_llm_checks_skip_the_port_of_another_machine(tmp_path):
    values = _local_values(LOCAL_LLM_BASE_URL="http://gpu-box:8080/v1")
    results = _by_name(check_local_llm(values, {"HF_HUB_CACHE": str(tmp_path)}, FakeSystem()))
    assert results["local server port"].status is Status.OK


def test_local_llm_checks_fail_on_an_unusable_base_url(tmp_path):
    values = _local_values(LOCAL_LLM_BASE_URL="localhost:8080")
    results = _by_name(check_local_llm(values, {"HF_HUB_CACHE": str(tmp_path)}, FakeSystem()))
    assert results["LOCAL_LLM_BASE_URL"].status is Status.FAIL


def test_doctor_reports_the_local_llm_checks(tmp_path):
    dotenv = COMPLETE.replace("LLM_PROVIDER=anthropic", "LLM_PROVIDER=local")
    results = run_doctor(_install(tmp_path, dotenv), {"HF_HUB_CACHE": str(tmp_path / "hub")}, FakeSystem(), None)
    assert {"mlx-lm", "local model", "local server port"} <= {r.name for r in results}
    assert exit_code(results) == 0  # warnings only: the server is optional until used
