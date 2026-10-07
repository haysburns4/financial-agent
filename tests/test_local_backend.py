"""LLM_PROVIDER=local: an OpenAI-compatible MLX server, behind the OpenAI backend.

A real `openai.AsyncOpenAI` client over `httpx.MockTransport`, so the SDK's
own SSE parsing and error types are exercised; only the server is fake.
"""
import json

import httpx
import openai
import pytest

from src.cli.checks import Status, check_local_server
from src.llm import LLMConnectionError, Message, MessageComplete, TextDelta, ToolCallDelta, ToolDef, collect
from src.llm.local_backend import PLACEHOLDER_API_KEY, LocalBackend, make_client

BASE = "http://mlx.test/v1"
MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def _sse(*chunks: dict) -> bytes:
    events = [f"data: {json.dumps({'id': 'c', 'object': 'chat.completion.chunk', 'created': 0, 'model': MODEL, **c})}\n\n"
              for c in chunks]
    return ("".join(events) + "data: [DONE]\n\n").encode()


def _text(content: str, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 32, "completion_tokens": 3, "total_tokens": 35}}


def _backend(handler) -> tuple[LocalBackend, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = openai.AsyncOpenAI(
        base_url=BASE, api_key=PLACEHOLDER_API_KEY, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(record)),
    )
    return LocalBackend(client, MODEL, BASE, 180), seen


def _stream(body: bytes):
    return lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


async def test_streams_text_and_usage_like_openai():
    backend, seen = _backend(_stream(_sse(_text("Hel"), _text("lo", "stop"), USAGE)))

    deltas = [d async for d in backend.stream(system="be brief", messages=[Message(role="user", text="hi")])]

    assert [d.text for d in deltas if isinstance(d, TextDelta)] == ["Hel", "lo"]
    final = deltas[-1]
    assert isinstance(final, MessageComplete)
    assert final.message.text == "Hello" and final.stop_reason == "end_turn"
    assert (final.usage.input_tokens, final.usage.output_tokens) == (32, 3)

    body = json.loads(seen[0].content)
    assert str(seen[0].url) == f"{BASE}/chat/completions"
    assert body["model"] == MODEL  # repo path passed through verbatim
    assert body["max_completion_tokens"] == 4096 and body["stream"] is True
    assert seen[0].headers["authorization"] == f"Bearer {PLACEHOLDER_API_KEY}"
    assert backend.provider == "local"


async def test_missing_usage_counts_as_zero():
    backend, _ = _backend(_stream(_sse(_text("ok", "stop"))))  # no usage chunk at all
    response = await collect(backend.stream(system="s", messages=[Message(role="user", text="hi")]))
    assert response.text == "ok"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (0, 0)


async def test_partial_usage_counts_missing_fields_as_zero():
    partial = {"choices": [], "usage": {"completion_tokens": 3}}
    backend, _ = _backend(_stream(_sse(_text("ok", "stop"), partial)))
    response = await collect(backend.stream(system="s", messages=[Message(role="user", text="hi")]))
    assert (response.usage.input_tokens, response.usage.output_tokens) == (0, 3)


async def test_tool_calls_in_mlx_single_chunk_form():
    # mlx_lm.server sends id, name and the whole arguments in one delta.
    call = {"choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [{
        "index": 0, "id": "t1", "type": "function",
        "function": {"name": "get_positions", "arguments": '{"account_id": "A1"}'},
    }]}}]}
    done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    backend, seen = _backend(_stream(_sse(call, done)))
    tool = ToolDef(name="get_positions", description="d", parameters={"type": "object", "properties": {}})

    deltas = [d async for d in backend.stream(system="s", messages=[Message(role="user", text="q")], tools=[tool])]

    assert any(isinstance(d, ToolCallDelta) and d.name == "get_positions" for d in deltas)
    final = deltas[-1]
    assert final.stop_reason == "tool_use"
    assert final.message.tool_calls[0].arguments == {"account_id": "A1"}
    assert json.loads(seen[0].content)["tools"][0]["function"]["name"] == "get_positions"


async def test_a_server_that_is_down_is_named_not_a_raw_connection_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    backend, _ = _backend(refuse)
    with pytest.raises(LLMConnectionError) as exc:
        await collect(backend.stream(system="s", messages=[Message(role="user", text="hi")]))

    message = str(exc.value)
    assert BASE in message and "appears to be down" in message
    assert "mlx_lm.server" in message


async def test_a_slow_server_points_at_the_timeout_setting():
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    backend, _ = _backend(stall)
    with pytest.raises(LLMConnectionError, match="did not answer within 180s.*LOCAL_LLM_TIMEOUT_SECONDS"):
        await collect(backend.stream(system="s", messages=[Message(role="user", text="hi")]))


def test_client_has_a_placeholder_key_a_long_timeout_and_no_retries():
    client = make_client(BASE, 180)
    assert client.api_key == PLACEHOLDER_API_KEY
    assert client.timeout == 180
    assert client.max_retries == 0
    assert str(client.base_url).rstrip("/") == BASE


def test_local_is_a_config_flip():
    from src.llm import build_backend

    backend = build_backend(MODEL, provider="local")
    assert isinstance(backend, LocalBackend)
    assert (backend.provider, backend.model) == ("local", MODEL)


# ---------- doctor / setup ----------


def _sync_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_doctor_sees_the_server_and_its_models():
    listing = {"object": "list", "data": [{"id": MODEL, "object": "model"}]}
    result = check_local_server(BASE, _sync_client(lambda r: httpx.Response(200, json=listing)))
    assert result.status is Status.OK
    assert MODEL in result.message


def test_doctor_reports_a_server_that_is_down():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    result = check_local_server(BASE, _sync_client(refuse))
    assert result.status is Status.FAIL
    assert BASE in result.message and "mlx_lm.server" in (result.fix or "")


# ---------- per-task routing ----------


def _settings(**overrides):
    from src.config import Settings

    return Settings(**{"_env_file": None, "ETRADE_CONSUMER_KEY": "k", "ETRADE_CONSUMER_SECRET": "s",
                       "LLM_PROVIDER": "anthropic", **overrides})


def test_tasks_fall_back_to_llm_provider():
    from src.llm import resolve_task

    config = _settings()
    assert resolve_task("chat", config) == ("anthropic", config.LLM_CHAT_MODEL)
    assert resolve_task("synthesizer", config) == ("anthropic", config.LLM_SYNTHESIS_MODEL)


def test_a_task_override_routes_just_that_task():
    from src.llm import resolve_task

    config = _settings(LLM_PROVIDER_CHAT="local", LOCAL_LLM_MODEL="mlx-community/Some-Model")
    assert resolve_task("chat", config) == ("local", "mlx-community/Some-Model")
    assert resolve_task("synthesizer", config) == ("anthropic", config.LLM_SYNTHESIS_MODEL)


def test_local_as_the_default_serves_every_task_its_one_model():
    from src.llm import resolve_task

    config = _settings(LLM_PROVIDER="local", LLM_PROVIDER_SYNTHESIZER="openai")
    assert resolve_task("chat", config) == ("local", config.LOCAL_LLM_MODEL)
    assert resolve_task("synthesizer", config) == ("openai", config.LLM_SYNTHESIS_MODEL)
