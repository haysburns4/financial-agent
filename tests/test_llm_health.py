"""LLM readiness: the local MLX server probe, /health's "llm" section, and the
startup log. The server is an `httpx.MockTransport`; nothing touches a network."""
import httpx
import pytest
from loguru import logger

from src.config import settings
from src.llm import health
from src.llm.health import LocalServerMonitor, llm_health, log_startup

BASE = "http://mlx.test/v1"
MODEL = "mlx-community/Qwen3-14B-4bit"


def _local(**overrides):
    return settings.model_copy(update={
        "LLM_PROVIDER": "local", "LOCAL_LLM_BASE_URL": BASE, "LOCAL_LLM_MODEL": MODEL,
        "LLM_PROVIDER_CHAT": None, "LLM_PROVIDER_SYNTHESIZER": None, **overrides,
    })


def _serving(*models: str, calls: list | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(200, json={"object": "list", "data": [{"id": m, "object": "model"} for m in models]})
    return httpx.MockTransport(handler)


def _down() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)
    return httpx.MockTransport(handler)


@pytest.fixture
def logs():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(f"{m.record['level'].name} {m.record['message']}"))
    yield lines
    logger.remove(sink)


# ---------- probe ----------


async def test_probe_lists_the_models_the_server_reports():
    calls: list[str] = []
    probe = await LocalServerMonitor(BASE, transport=_serving(MODEL, calls=calls)).probe()

    assert probe.reachable and probe.models_available == [MODEL]
    assert calls == [f"{BASE}/models"]


async def test_probe_reports_a_refused_connection_as_unreachable():
    probe = await LocalServerMonitor(BASE, transport=_down()).probe()

    assert not probe.reachable and probe.error == "ConnectError"


async def test_probe_is_cached_until_the_ttl_passes():
    calls: list[str] = []
    now = [0.0]
    monitor = LocalServerMonitor(BASE, ttl_seconds=30, transport=_serving(MODEL, calls=calls), clock=lambda: now[0])

    await monitor.probe()
    now[0] = 29.9
    await monitor.probe()
    assert len(calls) == 1

    now[0] = 30.0
    await monitor.probe()
    assert len(calls) == 2

    await monitor.probe(fresh=True)
    assert len(calls) == 3


# ---------- /health section ----------


async def test_health_omits_local_when_no_task_resolves_to_it():
    config = _local(LLM_PROVIDER="anthropic")
    monitor = LocalServerMonitor(BASE, transport=_down())

    section, alerts = await llm_health(monitor, config)

    assert section == {"providers": {"chat": "anthropic", "synthesizer": "anthropic"}}
    assert alerts == []


async def test_health_reports_a_reachable_local_server():
    section, alerts = await llm_health(LocalServerMonitor(BASE, transport=_serving(MODEL)), _local())

    local = section["local"]
    assert section["providers"] == {"chat": "local", "synthesizer": "local"}
    assert local["configured"] and local["reachable"]
    assert (local["base_url"], local["model"], local["models_available"]) == (BASE, MODEL, [MODEL])
    assert local["last_checked"].endswith("Z")
    assert alerts == []


async def test_health_alerts_when_a_task_routed_to_local_has_no_server():
    config = _local(LLM_PROVIDER="anthropic", LLM_PROVIDER_CHAT="local")

    section, alerts = await llm_health(LocalServerMonitor(BASE, transport=_down()), config)

    assert section["providers"] == {"chat": "local", "synthesizer": "anthropic"}
    assert section["local"]["reachable"] is False
    assert len(alerts) == 1 and BASE in alerts[0] and "chat" in alerts[0]


async def test_health_endpoint_is_degraded_not_failed_when_the_server_is_down(monkeypatch):
    from src import server

    monkeypatch.setattr(health, "settings", _local())
    monkeypatch.setattr(health, "MONITOR", LocalServerMonitor(BASE, transport=_down()))

    async def db_ok() -> bool:
        return True

    monkeypatch.setattr(server, "health_check", db_ok)
    transport = httpx.ASGITransport(app=server.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "degraded" and body["db"] is True
    assert body["llm"]["local"]["reachable"] is False
    assert body["alerts"] and BASE in body["alerts"][0]


# ---------- startup ----------


async def test_startup_warns_with_the_url_and_how_to_start_the_server(logs):
    summary = await log_startup(LocalServerMonitor(BASE, transport=_down()), _local())

    warning = next(line for line in logs if line.startswith("WARNING"))
    assert BASE in warning and "mlx_lm.server" in warning
    assert "unreachable" in summary


async def test_startup_warns_when_the_configured_model_is_not_served(logs):
    summary = await log_startup(LocalServerMonitor(BASE, transport=_serving("mlx-community/other")), _local())

    assert any(line.startswith("WARNING") and "LOCAL_LLM_MODEL" in line for line in logs)
    assert "reachable" in summary


async def test_startup_is_quiet_when_the_served_model_matches(logs):
    summary = await log_startup(LocalServerMonitor(BASE, transport=_serving(MODEL)), _local())

    assert not any(line.startswith("WARNING") for line in logs)
    assert summary == f"LLM chat=local, synthesizer=local; local server reachable at {BASE}"


async def test_startup_does_not_probe_without_a_local_task(logs):
    calls: list[str] = []
    monitor = LocalServerMonitor(BASE, transport=_serving(MODEL, calls=calls))

    summary = await log_startup(monitor, _local(LLM_PROVIDER="openai"))

    assert calls == [] and summary == "LLM chat=openai, synthesizer=openai"
