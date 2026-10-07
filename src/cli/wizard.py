"""`setup`: ask for what .env is missing, check it as it goes, and write it.

Only settings that are missing or invalid are asked for; `ask_all` re-asks the
whole flow with current values as defaults. Every answer is written to .env
straight away (atomically, 0600), so an interrupted run keeps its progress.
Secrets are never echoed: at most "set (…last4)".
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from cryptography.fernet import Fernet

from src.cli import checks
from src.cli.checks import CheckResult, NetworkChecks, Status, System, mask
from src.cli.doctor import env_values, local_checks
from src.cli.env_file import EnvFile, write_atomic
from src.cli.spec import (
    BY_NAME,
    OPENAI_SDK_PROVIDERS,
    PROVIDERS,
    PROVIDER_KEYS,
    PROVIDER_MODELS,
    SPEC,
    parse_bool,
    problem,
    providers_in_use,
)
from src.cli.ui import Prompter

SANDBOX = "Sandbox: E-Trade's test environment with fake data"
REAL = "Real account: your actual brokerage data (production keys)"
OPENAI_SYNC = ("uv", "sync", "--extra", "dev", "--extra", "openai")

# Asked in this order; anything else is only asked about when it is invalid.
FLOW = (
    "ETRADE_SANDBOX",
    "ETRADE_CONSUMER_KEY",
    "ETRADE_CONSUMER_SECRET",
    "LLM_PROVIDER",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_LLM_BASE_URL",
    "LOCAL_LLM_MODEL",
    "LLM_CHAT_MODEL",
    "LLM_SYNTHESIS_MODEL",
    "WATCHLIST",
)


class Wizard:
    def __init__(
        self,
        root: Path,
        prompter: Prompter,
        system: System,
        network: NetworkChecks | None,
        environ: Mapping[str, str],
        ask_all: bool = False,
    ) -> None:
        self.root = root
        self.io = prompter
        self.system = system
        self.network = network
        self.environ = environ
        self.ask_all = ask_all
        self.path = root / ".env"
        self.env = EnvFile([])

    # ---------- entry points ----------

    def bootstrap(self) -> None:
        """The parts that need no answers: .env, the token key, web/.env.local."""
        if self.path.exists():
            self.env = EnvFile.read(self.path)
        else:
            example = self.root / ".env.example"
            self.env = EnvFile.read(example)
            self.env.write(self.path)
            self.io.info("Created .env from .env.example")
            # Everything in a fresh copy is a template default, not a choice.
            self.ask_all = True
        self._token_key()
        self._web_env_file()

    def run(self) -> None:
        self.bootstrap()
        sandbox_changed = self._sandbox()
        self._etrade_keys(force=sandbox_changed)
        provider, provider_changed = self._provider()
        # LLM_PROVIDER plus any provider a task is routed to (set in .env).
        used = [provider, *sorted(providers_in_use(self._values()) - {provider})]
        self._provider_package(used)
        for name in used:
            self._provider_key(name)
        self._models(provider, provider_changed)
        self._watchlist()
        self._invalid_others()
        self._agui_url()
        self._summary()

    # ---------- state ----------

    def _values(self) -> dict[str, str]:
        return env_values(self.env)

    def _set(self, name: str, value: str) -> None:
        self.env.set(name, value)
        self.env.write(self.path)

    def _problem(self, name: str, value: str) -> str | None:
        return problem(BY_NAME[name], {**self._values(), name: value})

    def _needs(self, name: str) -> bool:
        value = self.env.get(name)
        return self.ask_all or not value or self._problem(name, value) is not None

    # ---------- prompts ----------

    def _ask_text(self, name: str, label: str, default: str) -> str:
        while True:
            answer = self.io.text(label, default=default).strip()
            reason = self._problem(name, answer)
            if reason is None:
                return answer
            self.io.info(f"{reason}; try again.")

    def _ask_secret(self, name: str, label: str) -> str:
        current = self.env.get(name) or ""
        keep = bool(current) and self._problem(name, current) is None
        hint = f" [{mask(current)}; Enter keeps it]" if keep else ""
        while True:
            answer = self.io.secret(f"{label}{hint}").strip()
            if not answer and keep:
                return current
            reason = self._problem(name, answer)
            if reason is None:
                return answer
            self.io.info(f"{reason}; try again.")

    def _credentials(
        self,
        fields: Sequence[tuple[str, str]],
        check: Callable[[Sequence[str]], CheckResult] | None,
        force: bool = False,
    ) -> None:
        """Ask for a set of credentials and re-ask straight away if the check rejects them.

        Credentials already in .env are checked too, and re-asked if rejected.
        """
        names = [name for name, _ in fields]
        needs = force or any(self._needs(name) for name in names)
        if not needs and check is not None:
            result = check([self.env.get(name) or "" for name in names])
            self.io.results([result])
            needs = result.status is Status.FAIL
        while needs:
            values = [self._ask_secret(name, label) for name, label in fields]
            result = check(values) if check is not None else None
            if result is not None:
                self.io.results([result])
            if (
                result is not None
                and result.status is Status.FAIL
                and self.io.confirm("Enter them again?", default=True)
            ):
                continue
            for name, value in zip(names, values):
                self._set(name, value)
            needs = False

    # ---------- steps ----------

    def _sandbox(self) -> bool:
        before = parse_bool(self.env.get("ETRADE_SANDBOX") or "true")
        if not self._needs("ETRADE_SANDBOX"):
            return False
        self.io.info(
            "E-Trade issues separate sandbox and production keys at developer.etrade.com. "
            "The sandbox needs only sandbox keys; your real account needs production keys, "
            "which E-Trade approves separately."
        )
        choice = self.io.select("Which E-Trade environment?", [SANDBOX, REAL], SANDBOX if before else REAL)
        sandbox = choice == SANDBOX
        self._set("ETRADE_SANDBOX", "true" if sandbox else "false")
        # Keys belong to one environment, so switching means new keys.
        return sandbox != before and bool(self.env.get("ETRADE_CONSUMER_KEY"))

    def _etrade_keys(self, force: bool) -> None:
        network = self.network
        check = None if network is None else (lambda v: network.etrade_keys(v[0], v[1]))
        self._credentials(
            [
                ("ETRADE_CONSUMER_KEY", "E-Trade consumer key"),
                ("ETRADE_CONSUMER_SECRET", "E-Trade consumer secret"),
            ],
            check,
            force=force,
        )

    def _provider(self) -> tuple[str, bool]:
        before = (self.env.get("LLM_PROVIDER") or BY_NAME["LLM_PROVIDER"].default or "").strip().lower()
        if not self._needs("LLM_PROVIDER"):
            return before, False
        default = before if before in PROVIDERS else PROVIDERS[0]
        provider = self.io.select("LLM provider", PROVIDERS, default)
        self._set("LLM_PROVIDER", provider)
        return provider, provider != before

    def _provider_package(self, providers: list[str]) -> None:
        needing = [p for p in providers if p in OPENAI_SDK_PROVIDERS]
        if not needing or self.system.module_available("openai"):
            return
        command = " ".join(OPENAI_SYNC)
        if not self.io.confirm(f"The openai package is not installed. Run `{command}` now?"):
            self.io.info(f"Provider {' and '.join(needing)} will not start until you run `{command}`.")
            return
        if self.system.run(OPENAI_SYNC, self.root) != 0:
            self.io.info(f"`{command}` failed; run it yourself to see why.")

    def _provider_key(self, provider: str) -> None:
        if provider == "local":
            self._local_server()
            return
        name = PROVIDER_KEYS[provider]
        network = self.network
        check = None if network is None else (lambda v: network.llm_key(provider, v[0]))
        self._credentials([(name, f"{provider} API key")], check)

    def _local_server(self) -> None:
        """The local provider has no key: ask where the MLX server is, then
        check it answers. A server that is down is reported, not re-asked —
        the URL is usually right and the server just isn't started yet."""
        name = "LOCAL_LLM_BASE_URL"
        url = self.env.get(name) or BY_NAME[name].default or ""
        if self._needs(name):
            url = self._ask_text(name, "MLX server URL (OpenAI-compatible, from mlx_lm.server)", url)
            self._set(name, url)
        if self.network is not None:
            self.io.results([self.network.local_server(url)])

    def _models(self, provider: str, provider_changed: bool) -> None:
        if provider == "local":
            # The server serves one model, for every task routed to it.
            name = "LOCAL_LLM_MODEL"
            if provider_changed or self._needs(name):
                current = self.env.get(name) or BY_NAME[name].default or ""
                self._set(name, self._ask_text(name, "Model the MLX server serves", current))
            return
        names = ("LLM_CHAT_MODEL", "LLM_SYNTHESIS_MODEL")
        if not (provider_changed or any(self._needs(name) for name in names)):
            return
        suggested = PROVIDER_MODELS[provider]
        labels = ("Chat model", "Synthesis model (runs on a schedule)")
        for name, label, fallback in zip(names, labels, suggested):
            current = self.env.get(name)
            usable = current and not provider_changed and self._problem(name, current) is None
            self._set(name, self._ask_text(name, label, current if usable else fallback))

    def _watchlist(self) -> None:
        if not self._needs("WATCHLIST"):
            return
        current = self.env.get("WATCHLIST")
        usable = current and self._problem("WATCHLIST", current) is None
        default = current if usable else BY_NAME["WATCHLIST"].default or ""
        answer = self._ask_text("WATCHLIST", "Watchlist (comma-separated tickers)", default)
        self._set("WATCHLIST", ",".join(t.strip().upper() for t in answer.split(",") if t.strip()))

    def _invalid_others(self) -> None:
        """Settings outside the flow are left alone unless they would break startup."""
        for spec in SPEC:
            if spec.name in FLOW or spec.name == "TOKEN_ENCRYPTION_KEY":
                continue
            value = self.env.get(spec.name)
            if value is None or self._problem(spec.name, value) is None:
                continue
            self.io.info(f"{self._problem(spec.name, value)}. {spec.help}")
            if spec.secret:
                self._set(spec.name, self._ask_secret(spec.name, spec.name))
            else:
                self._set(spec.name, self._ask_text(spec.name, spec.name, spec.default or ""))

    def _token_key(self) -> None:
        current = self.env.get("TOKEN_ENCRYPTION_KEY")
        if current and self._problem("TOKEN_ENCRYPTION_KEY", current) is None:
            return
        self._set("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
        self.io.info("Generated TOKEN_ENCRYPTION_KEY so the E-Trade login survives restarts")

    def _web_env_file(self) -> None:
        path = self.root / "web" / ".env.local"
        example = self.root / "web" / ".env.local.example"
        if path.exists() or not example.exists():
            return
        write_atomic(path, example.read_text(encoding="utf-8"))
        self.io.info("Created web/.env.local from web/.env.local.example")

    def _agui_url(self) -> None:
        path = self.root / "web" / ".env.local"
        result = checks.check_web_env(self.root / "web", self._values())
        if result.status is Status.OK or not path.exists():
            return
        expected = checks.expected_agui_url(self._values())
        if self.io.confirm(f"web/.env.local points AGUI_URL elsewhere. Change it to {expected}?"):
            web_env = EnvFile.read(path)
            web_env.set("AGUI_URL", expected)
            web_env.write(path)

    def _summary(self) -> None:
        self.io.info(f"Saved {self.path}")
        for name in FLOW:
            value = self.env.get(name)
            if value is None:
                continue
            shown = mask(value) if BY_NAME[name].secret else value
            self.io.info(f"  {name} = {shown}")
        self.io.results(local_checks(self.root, self._values(), self.environ, self.system))
