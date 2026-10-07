"""Setup checks: each returns a CheckResult saying ok / warn / fail and how to fix it.

Anything that touches the network or the machine goes through a seam — the
`NetworkChecks` and `System` protocols — so the wizard and doctor can be tested
with scripted fakes. No check ever puts a setting's value in its message.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import socket
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx
import pyetrade
import requests
from requests_oauthlib.oauth1_session import TokenRequestDenied

from src.cli.env_file import EnvFile
from src.cli.spec import BY_NAME, OPENAI_SDK_PROVIDERS, PROVIDER_KEYS, SPEC, problem, providers_in_use
from src.etrade.errors import denied

NETWORK_TIMEOUT = 10.0
MIN_NODE_MAJOR = 20
WEB_PORT = 3000
SETUP_HINT = "run `uv run python -m src.cli setup`"


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    fix: str | None = None


def ok(name: str, message: str) -> CheckResult:
    return CheckResult(name, Status.OK, message)


def warn(name: str, message: str, fix: str | None = None) -> CheckResult:
    return CheckResult(name, Status.WARN, message, fix)


def fail(name: str, message: str, fix: str | None = None) -> CheckResult:
    return CheckResult(name, Status.FAIL, message, fix)


def mask(value: str | None) -> str:
    """The most of a secret anyone gets to see."""
    if not value:
        return "not set"
    return f"set (…{value[-4:]})" if len(value) >= 16 else "set"


# ---------- network ----------


class NetworkChecks(Protocol):
    def llm_key(self, provider: str, key: str) -> CheckResult: ...

    def local_server(self, base_url: str) -> CheckResult: ...

    def etrade_keys(self, consumer_key: str, consumer_secret: str) -> CheckResult: ...


def check_llm_key(provider: str, key: str, client: httpx.Client) -> CheckResult:
    """List models: the cheapest call that proves the key authenticates."""
    name = f"{provider} API key"
    if provider == "anthropic":
        request = client.build_request(
            "GET",
            "https://api.anthropic.com/v1/models",
            params={"limit": 1},
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
    else:
        request = client.build_request(
            "GET", "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}
        )
    try:
        response = client.send(request)
    except httpx.HTTPError as exc:
        return warn(name, f"could not reach {provider}: {type(exc).__name__}", "check your connection")
    if response.status_code == 200:
        return ok(name, "accepted")
    if response.status_code in (401, 403):
        return fail(name, "key rejected", f"create a new key in the {provider} console")
    return warn(name, f"{provider} answered HTTP {response.status_code}", "try again later")


# What E-Trade's oauth_problem codes mean for someone running setup. The
# request-token endpoint is the same for sandbox and production keys, so a
# sandbox/production mix-up cannot be told apart from a wrong key here.
_ETRADE_PROBLEMS: Mapping[str, tuple[str, str]] = {
    "consumer_key_rejected": (
        "E-Trade does not recognise this consumer key",
        "copy it again from developer.etrade.com; newly issued production keys may not be active yet",
    ),
    "consumer_key_unknown": (
        "E-Trade does not recognise this consumer key",
        "copy it again from developer.etrade.com",
    ),
    "signature_invalid": (
        "the consumer secret does not match the key",
        "copy the secret that belongs to this key",
    ),
    "timestamp_refused": ("E-Trade refused the request time", "sync your system clock"),
}


def check_etrade_keys(
    consumer_key: str,
    consumer_secret: str,
    oauth_factory: Callable[[str, str], pyetrade.ETradeOAuth] = pyetrade.ETradeOAuth,
    timeout: float = NETWORK_TIMEOUT,
) -> CheckResult:
    """Ask E-Trade for a request token, which only needs the consumer key pair."""
    name = "E-Trade keys"
    # pyetrade sets no timeout of its own, so bound the wait from outside.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(lambda: oauth_factory(consumer_key, consumer_secret).get_request_token())
    try:
        future.result(timeout=timeout)
    except FutureTimeout:
        return warn(name, f"E-Trade did not answer within {timeout:.0f}s", "check your connection")
    except TokenRequestDenied as exc:
        code = denied(exc, "request token").oauth_problem
        message, fix = _ETRADE_PROBLEMS.get(
            code or "", (f"E-Trade rejected the keys ({code or 'no reason given'})", "check both values")
        )
        return fail(name, message, fix)
    except requests.RequestException as exc:
        return warn(name, f"could not reach E-Trade: {type(exc).__name__}", "check your connection")
    finally:
        pool.shutdown(wait=False)
    return ok(name, "accepted")


def check_local_server(base_url: str, client: httpx.Client) -> CheckResult:
    """LLM_PROVIDER=local: is the MLX server answering, and with which models?"""
    name = "local MLX server"
    start = "start it: `uv run mlx_lm.server --model <model> --port 8080`"
    try:
        response = client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        return fail(name, f"not reachable at {base_url} ({type(exc).__name__})", start)
    if response.status_code != 200:
        return fail(name, f"{base_url} answered HTTP {response.status_code}", "check LOCAL_LLM_BASE_URL")
    try:
        models = [m["id"] for m in response.json()["data"]]
    except (ValueError, KeyError, TypeError):
        return warn(name, f"{base_url} answered, but not with an OpenAI model list", "check LOCAL_LLM_BASE_URL")
    return ok(name, f"serving {', '.join(models) or 'no models'}")


class LiveNetworkChecks:
    def __init__(self, timeout: float = NETWORK_TIMEOUT) -> None:
        self._timeout = timeout

    def llm_key(self, provider: str, key: str) -> CheckResult:
        with httpx.Client(timeout=self._timeout) as client:
            return check_llm_key(provider, key, client)

    def etrade_keys(self, consumer_key: str, consumer_secret: str) -> CheckResult:
        return check_etrade_keys(consumer_key, consumer_secret, timeout=self._timeout)

    def local_server(self, base_url: str) -> CheckResult:
        with httpx.Client(timeout=self._timeout) as client:
            return check_local_server(base_url, client)


# ---------- machine ----------


@dataclass(frozen=True)
class Listener:
    pid: int
    command: str

    def __str__(self) -> str:
        return f"{self.command} (pid {self.pid})"


class System(Protocol):
    def node_version(self) -> str | None: ...

    def port_in_use(self, port: int) -> bool: ...

    def port_listener(self, port: int) -> Listener | None: ...

    def module_available(self, name: str) -> bool: ...

    def run(self, command: Sequence[str], cwd: Path) -> int: ...


class LiveSystem:
    def node_version(self) -> str | None:
        node = shutil.which("node")
        if node is None:
            return None
        result = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None

    def port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def port_listener(self, port: int) -> Listener | None:
        lsof = shutil.which("lsof")
        if lsof is None:
            return None
        result = subprocess.run(
            [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"],
            capture_output=True, text=True, timeout=10,
        )
        # -F output: one field per line, tagged by its first character.
        fields = {line[0]: line[1:] for line in result.stdout.splitlines() if line}
        if not fields.get("p", "").isdigit():
            return None
        return Listener(int(fields["p"]), fields.get("c", "unknown"))

    def module_available(self, name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def run(self, command: Sequence[str], cwd: Path) -> int:
        return subprocess.run(list(command), cwd=cwd).returncode


def check_node(system: System) -> CheckResult:
    name = "Node.js"
    version = system.node_version()
    fix = f"install Node {MIN_NODE_MAJOR} or newer (https://nodejs.org)"
    if version is None:
        return fail(name, "node is not on PATH", fix)
    match = re.match(r"v?(\d+)", version)
    if match is None or int(match.group(1)) < MIN_NODE_MAJOR:
        return fail(name, f"{version} is too old", fix)
    return ok(name, version)


def node_modules_stale(web: Path) -> bool:
    """True if web/node_modules is missing or older than package-lock.json."""
    modules, lock = web / "node_modules", web / "package-lock.json"
    if not modules.is_dir():
        return True
    # npm rewrites node_modules/.package-lock.json on every install.
    marker = modules / ".package-lock.json"
    installed = (marker if marker.exists() else modules).stat().st_mtime
    return lock.exists() and lock.stat().st_mtime > installed


def check_node_modules(web: Path) -> CheckResult:
    name = "web dependencies"
    fix = "run `cd web && npm install`"
    if not (web / "node_modules").is_dir():
        # A warning: ./start installs them itself.
        return warn(name, "web/node_modules is missing", "`./start` installs them, or run `cd web && npm ci`")
    if node_modules_stale(web):
        return warn(name, "package-lock.json changed since the last install", fix)
    return ok(name, "installed")


def check_port(port: int, system: System) -> CheckResult:
    name = f"port {port}"
    if not system.port_in_use(port):
        return ok(name, "free")
    holder = system.port_listener(port) or "another process"
    return warn(name, f"in use by {holder}", "stop it, or it may be the app already running")


def check_shadowing(env_values: Mapping[str, str], environ: Mapping[str, str]) -> list[CheckResult]:
    """Shell exports win over .env, so a stale one silently overrides setup."""
    results = []
    for spec in SPEC:
        if spec.name in environ and spec.name in env_values and environ[spec.name] != env_values[spec.name]:
            results.append(
                warn(
                    spec.name,
                    "exported in your shell; overrides .env",
                    f"run `unset {spec.name}` (and remove it from your shell profile)",
                )
            )
    return results


def check_settings(values: Mapping[str, str]) -> list[CheckResult]:
    problems = [p for p in (problem(spec, values) for spec in SPEC) if p]
    if not problems:
        return [ok("settings", "all valid")]
    return [fail("settings", p, SETUP_HINT) for p in problems]


def check_env_permissions(path: Path) -> CheckResult:
    name = ".env permissions"
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        return warn(name, "readable by other users", f"run `chmod 600 {path.name}`")
    return ok(name, "owner only")


def expected_agui_url(values: Mapping[str, str]) -> str:
    host = values.get("API_HOST") or BY_NAME["API_HOST"].default
    port = values.get("API_PORT") or BY_NAME["API_PORT"].default
    return f"http://{host}:{port}/agui"


def check_web_env(web: Path, values: Mapping[str, str]) -> CheckResult:
    name = "web/.env.local"
    path = web / ".env.local"
    if not path.exists():
        return warn(name, "missing", SETUP_HINT)
    expected = expected_agui_url(values)
    actual = EnvFile.read(path).get("AGUI_URL")
    if actual is not None and urlsplit(actual).netloc != urlsplit(expected).netloc:
        return warn(name, f"AGUI_URL does not point at the API ({expected})", SETUP_HINT)
    return ok(name, "present")


def check_provider_package(providers: set[str], system: System) -> CheckResult | None:
    """The openai package, if any provider in use needs it (openai, local)."""
    needing = sorted(providers & set(OPENAI_SDK_PROVIDERS))
    if not needing:
        return None
    name = "openai package"
    if system.module_available("openai"):
        return ok(name, "installed")
    return fail(name, f"provider {' and '.join(needing)} needs the openai package", "run `uv sync --extra dev --extra openai`")


def check_network(values: Mapping[str, str], network: NetworkChecks) -> list[CheckResult]:
    """Only checks credentials that are present; missing ones are check_settings' job."""
    results = []
    key, secret = values.get("ETRADE_CONSUMER_KEY"), values.get("ETRADE_CONSUMER_SECRET")
    if key and secret:
        results.append(network.etrade_keys(key, secret))
    for provider in sorted(providers_in_use(values)):
        if provider == "local":
            results.append(network.local_server(values.get("LOCAL_LLM_BASE_URL") or BY_NAME["LOCAL_LLM_BASE_URL"].default or ""))
        elif llm_key := values.get(PROVIDER_KEYS.get(provider, "")):
            results.append(network.llm_key(provider, llm_key))
    return results
