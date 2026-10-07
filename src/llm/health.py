"""LLM readiness: which provider each task resolves to, and — for the local
provider — whether the MLX server is up.

The failure to surface is a local provider configured with no server running:
without this, it shows only when a chat request hangs or fails. Nothing here
raises; pipelines, signals and the portfolio work without an LLM, so a down
server degrades the service rather than stopping it.

The probe is GET {LOCAL_LLM_BASE_URL}/models, cached for `ttl_seconds` so
/health polling from the web UI does not hammer the server.
"""
import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, get_args

import httpx
from loguru import logger

from src.config import Settings, settings
from src.llm.factory import Task, resolve_task

PROBE_TIMEOUT_SECONDS = 3.0
CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class LocalProbe:
    reachable: bool
    models_available: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None


class LocalServerMonitor:
    """Probes the MLX server, remembering the answer for `ttl_seconds`."""

    def __init__(
        self,
        base_url: str,
        *,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url
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
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.get(url)
        except httpx.HTTPError as exc:
            return LocalProbe(reachable=False, error=type(exc).__name__)
        if response.status_code != 200:
            return LocalProbe(reachable=False, error=f"HTTP {response.status_code}")
        try:
            models = [str(m["id"]) for m in response.json()["data"]]
        except (ValueError, KeyError, TypeError):
            return LocalProbe(reachable=False, error="not an OpenAI model list")
        return LocalProbe(reachable=True, models_available=models)


def task_providers(config: Settings = settings) -> dict[str, str]:
    """{task: provider}, as `backend_for` will resolve them."""
    return {task: resolve_task(task, config)[0] for task in get_args(Task)}


def start_hint(config: Settings = settings) -> str:
    return (
        f"start it with `uv run mlx_lm.server --model {config.LOCAL_LLM_MODEL} --port 8080`, "
        "or point LOCAL_LLM_BASE_URL at where it runs"
    )


async def llm_health(
    monitor: LocalServerMonitor, config: Settings | None = None
) -> tuple[dict[str, Any], list[str]]:  # anti-slop: allow no-any-returns - JSON body for /health
    """The /health "llm" section, and alerts for anything that degrades service.

    The "local" sub-object is present only when some task resolves to local.
    """
    config = config or settings
    providers = task_providers(config)
    section: dict[str, Any] = {"providers": providers}
    alerts: list[str] = []
    if "local" not in providers.values():
        return section, alerts

    probe = await monitor.probe()
    section["local"] = {
        "configured": True,
        "base_url": config.LOCAL_LLM_BASE_URL,
        "model": config.LOCAL_LLM_MODEL,
        "reachable": probe.reachable,
        "models_available": probe.models_available,
        "last_checked": probe.checked_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if not probe.reachable:
        tasks = ", ".join(t for t, p in providers.items() if p == "local")
        alerts.append(
            f"Local MLX server at {config.LOCAL_LLM_BASE_URL} is unreachable ({probe.error}); "
            f"LLM features ({tasks}) are down"
        )
    return section, alerts


async def log_startup(monitor: LocalServerMonitor, config: Settings = settings) -> str:
    """Probe at startup and log what was found; return a one-line readiness
    summary of the LLM layer. Never raises."""
    providers = task_providers(config)
    routing = ", ".join(f"{task}={provider}" for task, provider in providers.items())
    if "local" not in providers.values():
        return f"LLM {routing}"

    probe = await monitor.probe(fresh=True)
    if not probe.reachable:
        logger.warning(
            "Local MLX server at {} is not reachable ({}); LLM features are down until it is. {}",
            config.LOCAL_LLM_BASE_URL, probe.error, start_hint(config),
        )
        return f"LLM {routing}; local server unreachable at {config.LOCAL_LLM_BASE_URL}"

    served = probe.models_available
    logger.info("Local MLX server at {} serves: {}", config.LOCAL_LLM_BASE_URL, ", ".join(served) or "no models")
    if config.LOCAL_LLM_MODEL not in served:
        logger.warning(
            "LOCAL_LLM_MODEL={} is not among the models the MLX server reports ({}); "
            "requests may fail or load a different model",
            config.LOCAL_LLM_MODEL, ", ".join(served) or "none",
        )
    return f"LLM {routing}; local server reachable at {config.LOCAL_LLM_BASE_URL}"


MONITOR = LocalServerMonitor(settings.LOCAL_LLM_BASE_URL)
