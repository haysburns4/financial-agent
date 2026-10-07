"""`./start`: set up if needed, install, start the API and web UI, log in, supervise.

The API is only ever talked to over HTTP (src/cli must not import src.config).
"""
from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from src.cli import checks
from src.cli.checks import Listener, NetworkChecks, Status, System
from src.cli.doctor import effective_values, env_values
from src.cli.env_file import EnvFile
from src.cli.login import login
from src.cli.supervisor import Child, ProcessSpec, Supervisor
from src.cli.spec import BY_NAME
from src.cli.ui import Prompter
from src.cli.wizard import Wizard

API_SERVICE = "financial-agent"
WEB_SERVICE = "financial-agent-web"
WEB_HOST = "127.0.0.1"
API_START_TIMEOUT = 30.0
WEB_START_TIMEOUT = 120.0
# Files whose change means `next build` output is out of date, besides web/app/.
BUILD_INPUTS = ("package-lock.json", "package.json", "next.config.ts", "tsconfig.json")


@dataclass(frozen=True)
class Options:
    dev: bool = False
    open_browser: bool = True


class PortClaim(StrEnum):
    FREE = "free"
    REUSE = "reuse"
    BLOCKED = "blocked"


class _Health(BaseModel):
    service: str


def _files(directory: Path) -> Iterator[Path]:
    return (p for p in directory.rglob("*") if p.is_file())


def build_stale(web: Path) -> bool:
    """Whether `npm run build` is needed: no build yet, or a source is newer than it."""
    marker = web / ".next" / "BUILD_ID"
    if not marker.exists():
        return True
    built = marker.stat().st_mtime
    inputs = [*_files(web / "app"), *(web / name for name in BUILD_INPUTS)]
    return any(p.exists() and p.stat().st_mtime > built for p in inputs)


def service_at(url: str) -> str | None:
    """The `service` a health endpoint names, or None if nothing of ours answers."""
    try:
        response = httpx.get(url, timeout=1.0)
        return _Health.model_validate_json(response.content).service
    except (httpx.HTTPError, ValidationError):
        return None


class Launcher:
    def __init__(
        self,
        root: Path,
        prompter: Prompter,
        system: System,
        network: NetworkChecks | None,
        environ: Mapping[str, str],
        supervisor: Supervisor,
        open_url: Callable[[str], bool],
        interactive: bool,
    ) -> None:
        self.root = root
        self.web = root / "web"
        self.io = prompter
        self.system = system
        self.network = network
        self.environ = environ
        self.supervisor = supervisor
        self.open_url = open_url
        self.interactive = interactive

    def run(self, options: Options) -> int:
        try:
            return self._run(options)
        except KeyboardInterrupt:
            self.supervisor.say("\nStopping…")
            return 130
        finally:
            self.supervisor.stop()

    def _run(self, options: Options) -> int:
        values = self._preflight()
        if values is None or not self._install(values):
            return 1

        port = int(values.get("API_PORT") or BY_NAME["API_PORT"].default or "8000")
        api_url = f"http://{values.get('API_HOST') or BY_NAME['API_HOST'].default}:{port}"
        web_url = f"http://{WEB_HOST}:{checks.WEB_PORT}"

        # Both ports before starting anything, so a blocked one fails fast.
        api_claim = self._claim(port, f"{api_url}/health", API_SERVICE, "API")
        if api_claim is PortClaim.BLOCKED:
            return 1
        web_claim = self._claim(checks.WEB_PORT, f"{web_url}/api/health", WEB_SERVICE, "web UI")
        if web_claim is PortClaim.BLOCKED:
            return 1

        if api_claim is PortClaim.FREE and not self._start_api(api_url):
            return 1
        self._login(api_url)
        if web_claim is PortClaim.FREE and not self._start_web(web_url, options):
            return 1

        self.io.info(f"financial-agent is running at {web_url}")
        if options.open_browser:
            self.open_url(web_url)
        if not self.supervisor.children:
            self.io.info("Both parts were already running; nothing left for this terminal to do.")
            return 0
        self.io.info("Press Ctrl-C to stop.")
        self.supervisor.echo_all()
        return self.supervisor.wait()

    # ---------- 1. preflight ----------

    def _values(self) -> dict[str, str]:
        return effective_values(env_values(EnvFile.read(self.root / ".env")), self.environ)

    def _settings_fail(self) -> list[checks.CheckResult]:
        if not (self.root / ".env").exists():
            return [checks.fail(".env", "not found", checks.SETUP_HINT)]
        return [r for r in checks.check_settings(self._values()) if r.status is Status.FAIL]

    def _preflight(self) -> dict[str, str] | None:
        failures = self._settings_fail()
        if failures:
            if not self.interactive:
                self.io.results(failures)
                self.io.info("Run `./start setup` in a terminal first.")
                return None
            self.io.info("Some settings are missing; starting setup.")
            Wizard(self.root, self.io, self.system, self.network, self.environ).run()
            failures = self._settings_fail()
            if failures:
                self.io.results(failures)
                return None

        node = checks.check_node(self.system)
        if node.status is Status.FAIL:
            self.io.results([node])
            return None
        env = env_values(EnvFile.read(self.root / ".env"))
        self.io.results(checks.check_shadowing(env, self.environ))
        return self._values()

    # ---------- 2. dependencies ----------

    def _install(self, values: Mapping[str, str]) -> bool:
        provider = (values.get("LLM_PROVIDER") or "").strip().lower()
        # --inexact keeps what else is installed (pytest, say) instead of pruning it.
        command = ["uv", "sync", "--inexact", "--quiet"]
        if provider == "openai":
            command += ["--extra", "openai"]
        if self.system.run(command, self.root) != 0:
            self.io.info(f"`{' '.join(command)}` failed.")
            return False
        if checks.node_modules_stale(self.web):
            self.io.info("Installing web dependencies (npm ci)…")
            if self.system.run(["npm", "ci"], self.web) != 0:
                self.io.info("`npm ci` failed in web/.")
                return False
        return True

    # ---------- 3. ports ----------

    def _claim(self, port: int, health_url: str, service: str, label: str) -> PortClaim:
        if not self.system.port_in_use(port):
            return PortClaim.FREE
        listener = self.system.port_listener(port)
        if service_at(health_url) != service:
            holder = listener or "another process"
            self.io.info(f"Port {port} is in use by {holder}. Stop it and run ./start again.")
            return PortClaim.BLOCKED
        if not self.interactive:
            self.io.info(f"Reusing the {label} already running on port {port}.")
            return PortClaim.REUSE

        reuse, restart = "Reuse it", "Stop it and start a fresh one"
        choice = self.io.select(
            f"financial-agent's {label} is already running on port {port}.",
            [reuse, restart, "Quit"],
            reuse,
        )
        if choice == reuse:
            return PortClaim.REUSE
        if choice == restart and listener is not None and self._stop_listener(port, listener):
            return PortClaim.FREE
        return PortClaim.BLOCKED

    def _stop_listener(self, port: int, listener: Listener) -> bool:
        for sig, wait in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 3.0)):
            try:
                os.kill(listener.pid, sig)
            except ProcessLookupError:
                return True
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if not self.system.port_in_use(port):
                    return True
                time.sleep(0.2)
        self.io.info(f"Could not stop {listener}.")
        return False

    # ---------- 4. API ----------

    def _start_api(self, api_url: str) -> bool:
        self.io.info("Starting the API…")
        child = self.supervisor.start(
            ProcessSpec(
                "api",
                ("uv", "run", "python", "-m", "src.main"),
                self.root,
                color="36",
                env={**self.environ, "PYTHONUNBUFFERED": "1"},
            ),
            echo=False,  # quiet until the login prompts are done
        )
        return self._wait_until_up(child, f"{api_url}/health", API_SERVICE, API_START_TIMEOUT)

    def _wait_until_up(self, child: Child, url: str, service: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if service_at(url) == service:
                return True
            if not child.running:
                self.io.info(f"{child.spec.name} exited with code {child.process.returncode} while starting:")
                self.supervisor.print_tail(child)
                return False
            time.sleep(0.3)
        self.io.info(f"{child.spec.name} did not come up within {timeout:.0f}s. Last lines:")
        self.supervisor.print_tail(child)
        return False

    # ---------- 5. E-Trade login ----------

    def _login(self, api_url: str) -> None:
        if not self.interactive:
            self.io.info("Not a terminal, so no E-Trade login; run `./start login` later.")
            return
        try:
            with httpx.Client(base_url=api_url, timeout=30.0) as client:
                login(client, self.io, self.open_url)
        except httpx.HTTPError as exc:
            self.io.info(f"E-Trade login failed talking to the API ({type(exc).__name__}); try `./start login`.")

    # ---------- 6. web ----------

    def _start_web(self, web_url: str, options: Options) -> bool:
        if options.dev:
            script = "dev"
        else:
            script = "start"
            if build_stale(self.web):
                self.io.info("Building the web UI (first run, or the code changed)…")
                build = ProcessSpec("web", ("npm", "run", "build"), self.web, color="35")
                if self.supervisor.run_to_completion(build) != 0:
                    self.io.info("`npm run build` failed; see the output above.")
                    return False
        self.io.info(f"Starting the web UI ({script})…")
        child = self.supervisor.start(
            ProcessSpec("web", ("npm", "run", script), self.web, color="35"), echo=False
        )
        return self._wait_until_up(child, f"{web_url}/api/health", WEB_SERVICE, WEB_START_TIMEOUT)
