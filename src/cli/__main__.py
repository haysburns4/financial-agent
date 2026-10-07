"""Setup commands.

    uv run python -m src.cli setup              # ask for whatever .env is missing
    uv run python -m src.cli setup --all        # re-ask everything
    uv run python -m src.cli doctor             # check the install, exit 1 on any failure

`--offline` skips the calls to E-Trade and the LLM provider. `--non-interactive`
never prompts: `setup` then only writes what needs no answers (.env from the
template, TOKEN_ENCRYPTION_KEY, web/.env.local) and reports like `doctor`.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from src.cli.checks import LiveNetworkChecks, LiveSystem
from src.cli.doctor import exit_code, run_doctor
from src.cli.ui import TerminalPrompter, results_table
from src.cli.wizard import Wizard

ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--offline", action="store_true", help="skip checks that call E-Trade or the LLM provider")
    common.add_argument("--non-interactive", action="store_true", help="never prompt (for CI)")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", parents=[common], help="write .env interactively")
    setup.add_argument("--all", action="store_true", help="re-ask every setting, not just missing ones")
    commands.add_parser("doctor", parents=[common], help="check the install without changing anything")
    return parser


def _report(console: Console, root: Path, offline: bool) -> int:
    network = None if offline else LiveNetworkChecks()
    results = run_doctor(root, os.environ, LiveSystem(), network)
    console.print(results_table(results))
    code = exit_code(results)
    console.print("[red]Some checks failed.[/red]" if code else "[green]No failures.[/green]")
    return code


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    console = Console()
    if args.command == "doctor":
        return _report(console, ROOT, args.offline)

    network = None if args.offline else LiveNetworkChecks()
    wizard = Wizard(ROOT, TerminalPrompter(console), LiveSystem(), network, os.environ, ask_all=args.all)
    if args.non_interactive:
        wizard.bootstrap()
        return _report(console, ROOT, args.offline)
    if not sys.stdin.isatty():
        console.print("setup needs a terminal to ask questions; use --non-interactive in scripts.")
        return 2
    try:
        wizard.run()
    except KeyboardInterrupt:
        console.print("\nCancelled. Answers given so far are saved in .env.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
