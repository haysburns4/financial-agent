"""LLM readiness: which provider each task resolves to, and — for the local
provider — whether its server is up and how large a context it allows.

The failure to surface is a local provider configured with no server running:
without this, it shows only when a chat request hangs or fails. The second is
a server whose context window is smaller than the prompt budget, which truncates
prompts silently. Nothing here raises; pipelines, signals and the portfolio work
without an LLM, so a down server degrades the service rather than stopping it.

The probe is GET {LOCAL_LLM_BASE_URL}/models, plus the server's own window
endpoint when it has one (src/llm/local_profiles.py), cached for
`ttl_seconds` so /health polling and chat turns do not hammer the server.
"""
import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, get_args

import httpx
from loguru import logger

from src.config import Settings, settings
from src.llm.factory import Task, resolve_task
from src.llm.local_profiles import ServerProfile, server_profile, server_root, start_hint

PROBE_TIMEOUT_SECONDS = 3.0
CACHE_TTL_SECONDS = 30.0
# Key in `context_windows` for a window that applies whatever the model id.
ANY_MODEL = ""


@dataclass(frozen=True)
class LocalProbe:
    reachable: bool
    models_available: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
    # Tokens of context the server allocated, by model id (or ANY_MODEL).
    # Empty when the server does not say or the model is not loaded yet.
    context_windows: dict[str, int] = field(default_factory=dict)

    def window_for(self, model: str) -> int | None:
        windows = self.context_windows
        return windows.get(model) or windows.get(_with_tag(model)) or windows.get(ANY_MODEL)


def _with_tag(model: str) -> str:
    """Ollama's name for an untagged model."""
    return model if ":" in model else f"{model}:latest"


class LocalServerMonitor:
    """Probes a local LLM server, remembering the answer for `ttl_seconds`."""

    def __init__(
        self,
        base_url: str,
        *,
        server: str | None = None,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url
        self.server: ServerProfile = server_profile(server)
        self._ttl = ttl_seconds
        self._timeout = timeout_seconds
        self._transport = transport
        self._clock = clock
        self._cached: tuple[float, LocalProbe] | None = None
        # One probe at a time: concurrent /health polls share it.
        self._lock = asyncio.Lock()

    async def probe(self, *, fresh: bool = False) -> LocalProbe:
        async with self._lock:
            if not fresh and self._cached is not None:
                at, result = self._cached
                if self._clock() - at < self._ttl:
                    return result
            result = await self._probe()
            self._cached = (self._clock(), result)
            return result

    async def _probe(self) -> LocalProbe:
        url = f"{self.base_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            try:
                response = await client.get(url)
            except httpx.HTTPError as exc:
                return LocalProbe(reachable=False, error=type(exc).__name__)
            if response.status_code != 200:
                return LocalProbe(reachable=False, error=f"HTTP {response.status_code}")
            try:
                listing = response.json()["data"]
                models = [str(m["id"]) for m in listing]
            except (ValueError, KeyError, TypeError):
                return LocalProbe(reachable=False, error="not an OpenAI model list")
            windows = await self._windows(client, listing)
        return LocalProbe(reachable=True, models_available=models, context_windows=windows)

    async def _windows(self, client: httpx.AsyncClient, listing: list[Any]) -> dict[str, int]:  # anti-slop: allow no-any-parameters - JSON from the server, validated field by field below
        """The context windows the server reports."""
        root = server_root(self.base_url)
        try:
            match self.server.window_source:
                case "vllm_models":
                    pairs = [(m["id"], m.get("max_model_len")) for m in listing]
                case "ollama_ps":
                    running = (await client.get(f"{root}/api/ps")).json()["models"]
                    pairs = [(m.get(key), m.get("context_length")) for m in running for key in ("model", "name")]
                case "llamacpp_props":
                    props = (await client.get(f"{root}/props")).json()
                    pairs = [(ANY_MODEL, props["default_generation_settings"]["n_ctx"])]
                case "lmstudio_models":
                    loaded = (await client.get(f"{root}/api/v0/models")).json()["data"]
                    pairs = [(m["id"], m.get("loaded_context_length")) for m in loaded]
                case None:
                    return {}
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.debug("local LLM: no context window from {} ({})", self.server.label, exc)
            return {}
        return {str(k): v for k, v in pairs if k is not None and isinstance(v, int) and v > 0}


_MONITORS: dict[tuple[str, str], LocalServerMonitor] = {}


def monitor_for(base_url: str, server: str | None) -> LocalServerMonitor:
    """The one monitor for this server, shared by /health and its backends."""
    key = (base_url, server_profile(server).kind)
    if key not in _MONITORS:
        _MONITORS[key] = LocalServerMonitor(base_url, server=server)
    return _MONITORS[key]


def task_providers(config: Settings = settings) -> dict[str, str]:
    """{task: provider}, as `backend_for` will resolve them."""
    return {task: resolve_task(task, config)[0] for task in get_args(Task)}


def local_models(config: Settings = settings) -> dict[str, str]:
    """{task: model} for the tasks that run on the local provider."""
    resolved = {task: resolve_task(task, config) for task in get_args(Task)}
    return {task: model for task, (provider, model) in resolved.items() if provider == "local"}


def window_problem(probe: LocalProbe, task: str, model: str, config: Settings, reply_tokens: int) -> str | None:
    """Why `task`'s prompt budget will be cut, or None if it fits."""
    window = probe.window_for(model)
    budget = config.LOCAL_LLM_CONTEXT_BUDGET
    if window is None or budget + reply_tokens <= window:
        return None
    server = server_profile(config.LOCAL_LLM_SERVER)
    fix = f"; to give it more, {server.window_hint}" if server.window_hint else ""
    reply = f" plus up to {reply_tokens} reply tokens" if reply_tokens else ""
    return (
        f"LOCAL_LLM_CONTEXT_BUDGET={budget}{reply} does not fit the {window}-token context window "
        f"{server.label} reports for {model}; {task} prompts are cut to {max(0, window - reply_tokens)}, "
        f"leaving less room for memory and tool results{fix}"
    )


def _window_problems(probe: LocalProbe, models: Mapping[str, str], config: Settings, reply_tokens: Mapping[str, int]) -> list[str]:
    return [
        p for task, model in models.items()
        if (p := window_problem(probe, task, model, config, reply_tokens.get(task, 0)))
    ]


async def llm_health(
    monitor: LocalServerMonitor, config: Settings | None = None, reply_tokens: Mapping[str, int] | None = None
) -> tuple[dict[str, Any], list[str]]:  # anti-slop: allow no-any-returns - JSON body for /health
    """The /health "llm" section, and alerts for anything that degrades service."""
    config = config or settings
    providers = task_providers(config)
    section: dict[str, Any] = {"providers": providers}
    alerts: list[str] = []
    models = local_models(config)
    if not models:
        return section, alerts

    probe = await monitor.probe()
    section["local"] = {
        "configured": True,
        "server": monitor.server.kind,
        "base_url": config.LOCAL_LLM_BASE_URL,
        "model": config.LOCAL_LLM_MODEL,
        "models": models,
        "reachable": probe.reachable,
        "models_available": probe.models_available,
        "context_windows": {m: probe.window_for(m) for m in sorted(set(models.values()))},
        "context_budget": config.LOCAL_LLM_CONTEXT_BUDGET,
        "last_checked": probe.checked_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if not probe.reachable:
        tasks = ", ".join(models)
        alerts.append(
            f"Local LLM server ({monitor.server.label}) at {config.LOCAL_LLM_BASE_URL} is unreachable "
            f"({probe.error}); LLM features ({tasks}) are down"
        )
    alerts.extend(_window_problems(probe, models, config, reply_tokens or {}))
    return section, alerts


async def log_startup(
    monitor: LocalServerMonitor, config: Settings = settings, reply_tokens: Mapping[str, int] | None = None
) -> str:
    """Probe at startup and log what was found."""
    providers = task_providers(config)
    routing = ", ".join(f"{task}={provider}" for task, provider in providers.items())
    models = local_models(config)
    if not models:
        return f"LLM {routing}"

    label = monitor.server.label
    probe = await monitor.probe(fresh=True)
    if not probe.reachable:
        logger.warning(
            "Local LLM server ({}) at {} is not reachable ({}); LLM features are down until it is. {}",
            label, config.LOCAL_LLM_BASE_URL, probe.error,
            start_hint(monitor.server, config.LOCAL_LLM_BASE_URL, config.LOCAL_LLM_MODEL),
        )
        return f"LLM {routing}; local server unreachable at {config.LOCAL_LLM_BASE_URL}"

    served = probe.models_available
    logger.info("Local LLM server ({}) at {} serves: {}", label, config.LOCAL_LLM_BASE_URL, ", ".join(served) or "no models")
    for task, model in models.items():
        if model not in served and _with_tag(model) not in served:
            setting = "LOCAL_LLM_MODEL" if model == config.LOCAL_LLM_MODEL else f"LOCAL_LLM_MODEL_{task.upper()}"
            logger.warning(
                "{}={} is not among the models the server reports ({}); "
                "requests may fail or load a different model",
                setting, model, ", ".join(served) or "none",
            )
    for problem in _window_problems(probe, models, config, reply_tokens or {}):
        logger.warning(problem)
    return f"LLM {routing}; local server reachable at {config.LOCAL_LLM_BASE_URL}"


MONITOR = monitor_for(settings.LOCAL_LLM_BASE_URL, settings.LOCAL_LLM_SERVER)
