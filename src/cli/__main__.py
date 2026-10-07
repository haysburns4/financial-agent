"""Launcher and setup commands. `./start` runs this; `start` is the default.

    ./start                  # set up if needed, then run the API and web UI
    ./start --dev            # same, with `next dev` instead of a production build
    ./start login            # E-Trade login against a running API (daily); or use the
                             # "Log in to E-Trade" button in the dashboard
    ./start setup [--all]    # ask for whatever .env is missing (or everything)
    ./start doctor           # check the install, exit 1 on any failure

`--offline` skips the calls to E-Trade and the LLM provider. `--non-interactive`
never prompts: `setup` then only writes what needs no answers (.env from the
template, TOKEN_ENCRYPTION_KEY, web/.env.local) and reports like `doctor`.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

import httpx
from rich.console import Console

from src.cli.checks import LiveNetworkChecks, LiveSystem
from src.cli.doctor import effective_values, env_values, exit_code, run_doctor
from src.cli.env_file import EnvFile
from src.cli.launcher import API_SERVICE, Launcher, Options, service_at
from src.cli.login import login
from src.cli.spec import BY_NAME
from src.cli.supervisor import Supervisor
from src.cli.ui import TerminalPrompter, results_table
from src.cli.wizard import Wizard

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = ("start", "login", "setup", "doctor")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="./start")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--offline", action="store_true", help="skip checks that call E-Trade or the LLM provider")
    common.add_argument("--non-interactive", action="store_true", help="never prompt (for CI)")
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", parents=[common], help="run the app (default)")
    start.add_argument("--dev", action="store_true", help="run the web UI with `next dev`")
    start.add_argument("--no-browser", action="store_true", help="don't open the dashboard")
    login_cmd = commands.add_parser(
        "login",
        help="log in to E-Trade against the running API (the dashboard's \"Log in to E-Trade\" button does the same)",
    )
    login_cmd.set_defaults(offline=False, non_interactive=False)
    setup = commands.add_parser("setup", parents=[common], help="write .env interactively")
    setup.add_argument("--all", action="store_true", help="re-ask every setting, not just missing ones")
    commands.add_parser("doctor", parents=[common], help="check the install without changing anything")
    return parser


def _with_default_command(argv: Sequence[str]) -> list[str]:
    if argv and (argv[0] in COMMANDS or argv[0] in ("-h", "--help")):
        return list(argv)
    return ["start", *argv]


def _report(console: Console, offline: bool) -> int:
    network = None if offline else LiveNetworkChecks()
    results = run_doctor(ROOT, os.environ, LiveSystem(), network)
    console.print(results_table(results))
    code = exit_code(results)
    console.print("[red]Some checks failed.[/red]" if code else "[green]No failures.[/green]")
    return code


def _login(prompter: TerminalPrompter) -> int:
    values = effective_values(env_values(EnvFile.read(ROOT / ".env")), os.environ)
    host = values.get("API_HOST") or BY_NAME["API_HOST"].default
    port = values.get("API_PORT") or BY_NAME["API_PORT"].default
    api_url = f"http://{host}:{port}"
    if service_at(f"{api_url}/health") != API_SERVICE:
        prompter.info(f"The API is not running at {api_url}. Start everything with `./start`.")
        return 1
    with httpx.Client(base_url=api_url, timeout=30.0) as client:
        login(client, prompter, webbrowser.open)
    return 0


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    # The children run in their own sessions, so a closed terminal (SIGHUP) or a
    # plain `kill` never reaches them; treat both like Ctrl-C and shut them down.
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    args = _parser().parse_args(_with_default_command(sys.argv[1:] if argv is None else argv))
    console = Console()
    prompter = TerminalPrompter(console)
    interactive = sys.stdin.isatty() and not args.non_interactive
    network = None if args.offline else LiveNetworkChecks()

    try:
        match args.command:
            case "doctor":
                return _report(console, args.offline)
            case "login":
                if not interactive:
                    console.print("login needs a terminal to ask for the verification code.")
                    return 2
                return _login(prompter)
            case "start":
                launcher = Launcher(
                    ROOT, prompter, LiveSystem(), network, os.environ,
                    Supervisor(), webbrowser.open, interactive,
                )
                return launcher.run(Options(dev=args.dev, open_browser=not args.no_browser))
            case _:
                wizard = Wizard(ROOT, prompter, LiveSystem(), network, os.environ, ask_all=args.all)
                if args.non_interactive:
                    wizard.bootstrap()
                    return _report(console, args.offline)
                if not interactive:
                    console.print("setup needs a terminal to ask questions; use --non-interactive in scripts.")
                    return 2
                wizard.run()
                return 0
    except KeyboardInterrupt:
        # setup saves each answer as it goes, so nothing given so far is lost.
        console.print("\nCancelled.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
