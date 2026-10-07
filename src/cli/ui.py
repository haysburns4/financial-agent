"""Terminal IO behind the `Prompter` protocol, so tests can script the answers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import questionary
from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.cli.checks import CheckResult, Status

_STYLES = {Status.OK: "green", Status.WARN: "yellow", Status.FAIL: "red"}


class Prompter(Protocol):
    def text(self, message: str, default: str = "") -> str: ...

    def secret(self, message: str) -> str: ...

    def select(self, message: str, choices: Sequence[str], default: str) -> str: ...

    def confirm(self, message: str, default: bool = True) -> bool: ...

    def info(self, message: str) -> None: ...

    def results(self, results: Sequence[CheckResult]) -> None: ...


def results_table(results: Sequence[CheckResult]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Fix")
    for r in results:
        # Text(), not markup: messages can contain brackets.
        table.add_row(
            Text(r.status.value.upper(), style=_STYLES[r.status]),
            Text(r.name),
            Text(r.message),
            Text(r.fix or ""),
        )
    return table


class TerminalPrompter:
    """questionary for input, rich for output. Ctrl-C raises KeyboardInterrupt."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def text(self, message: str, default: str = "") -> str:
        return questionary.text(message, default=default).unsafe_ask()

    def secret(self, message: str) -> str:
        return questionary.password(message).unsafe_ask()

    def select(self, message: str, choices: Sequence[str], default: str) -> str:
        return questionary.select(message, choices=list(choices), default=default).unsafe_ask()

    def confirm(self, message: str, default: bool = True) -> bool:
        return questionary.confirm(message, default=default).unsafe_ask()

    def info(self, message: str) -> None:
        self.console.print(Text(message))

    def results(self, results: Sequence[CheckResult]) -> None:
        for r in results:
            line = Text(f"{r.status.value.upper():<5}", style=_STYLES[r.status])
            line.append(f" {r.name}: {r.message}")
            if r.fix and r.status is not Status.OK:
                line.append(f" — {r.fix}", style="dim")
            self.console.print(line)
