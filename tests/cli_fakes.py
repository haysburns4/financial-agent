"""Scripted stand-ins for the wizard's IO, network and machine seams."""
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from src.cli.checks import CheckResult, Status, ok

# A .env the wizard and doctor have nothing to say about.
ETRADE_KEY = "etrade-key-0000000000001234"
ETRADE_SECRET = "etrade-secret-00000000005678"
ANTHROPIC_KEY = "sk-ant-test-00000000000000abcd"
TOKEN_KEY = "kFhzNZQ8d6P8rH2bKj4C9H3X3PZ2u3mGm3bqgM5H6nU="

COMPLETE = f"""\
ETRADE_SANDBOX=true
ETRADE_CONSUMER_KEY={ETRADE_KEY}
ETRADE_CONSUMER_SECRET={ETRADE_SECRET}
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY={ANTHROPIC_KEY}
LLM_CHAT_MODEL=claude-opus-5
LLM_SYNTHESIS_MODEL=claude-sonnet-5
WATCHLIST=AAPL,MSFT
TOKEN_ENCRYPTION_KEY={TOKEN_KEY}
"""


class ScriptedPrompter:
    """Answers prompts in order; each script entry is (text the prompt must contain, answer).

    Confirm answers are "y" / "n".
    """

    def __init__(self, script: Sequence[tuple[str, str]] = ()) -> None:
        self.script = deque(script)
        self.asked: list[str] = []
        self.output: list[str] = []

    def _answer(self, message: str) -> str:
        self.asked.append(message)
        assert self.script, f"unexpected prompt: {message!r}"
        expected, answer = self.script.popleft()
        assert expected in message, f"expected a prompt about {expected!r}, got {message!r}"
        return answer

    def text(self, message: str, default: str = "") -> str:
        answer = self._answer(message)
        return default if answer == "<default>" else answer

    def secret(self, message: str) -> str:
        return self._answer(message)

    def select(self, message: str, choices: Sequence[str], default: str) -> str:
        answer = self._answer(message)
        matches = [c for c in choices if answer in c]
        assert len(matches) == 1, f"{answer!r} picks {matches} from {choices}"
        return matches[0]

    def confirm(self, message: str, default: bool = True) -> bool:
        return self._answer(message) == "y"

    def info(self, message: str) -> None:
        self.output.append(message)

    def results(self, results: Sequence[CheckResult]) -> None:
        self.output.extend(f"{r.status} {r.name}: {r.message} {r.fix or ''}" for r in results)


class FakeNetwork:
    """Replays results per check, then answers ok; records what it was asked."""

    def __init__(
        self, etrade: Sequence[CheckResult] = (), llm: Sequence[CheckResult] = ()
    ) -> None:
        self._etrade = deque(etrade)
        self._llm = deque(llm)
        self.etrade_calls: list[tuple[str, str]] = []
        self.llm_calls: list[tuple[str, str]] = []

    def llm_key(self, provider: str, key: str) -> CheckResult:
        self.llm_calls.append((provider, key))
        return self._llm.popleft() if self._llm else ok(f"{provider} API key", "accepted")

    def etrade_keys(self, consumer_key: str, consumer_secret: str) -> CheckResult:
        self.etrade_calls.append((consumer_key, consumer_secret))
        return self._etrade.popleft() if self._etrade else ok("E-Trade keys", "accepted")


class FakeSystem:
    def __init__(
        self,
        node: str | None = "v22.1.0",
        busy_ports: Sequence[int] = (),
        modules: Sequence[str] = ("openai",),
        run_code: int = 0,
    ) -> None:
        self.node = node
        self.busy_ports = set(busy_ports)
        self.modules = set(modules)
        self.run_code = run_code
        self.commands: list[tuple[str, ...]] = []

    def node_version(self) -> str | None:
        return self.node

    def port_in_use(self, port: int) -> bool:
        return port in self.busy_ports

    def port_holder(self, port: int) -> str | None:
        return "python3 (pid 4242)" if port in self.busy_ports else None

    def module_available(self, name: str) -> bool:
        return name in self.modules

    def run(self, command: Sequence[str], cwd: Path) -> int:
        self.commands.append(tuple(command))
        return self.run_code


def is_fail(result: CheckResult) -> bool:
    return result.status is Status.FAIL
