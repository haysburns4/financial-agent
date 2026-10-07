"""Guided E-Trade login against a running API (`./start login`).

E-Trade ends every session at midnight ET, so this runs daily. It drives the
API's own OAuth endpoints over HTTP rather than importing src.etrade.auth,
which would pull in src.config. The dashboard's "Log in to E-Trade" button
(web/app/etrade-login.tsx) drives the same endpoints from the browser.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import httpx
from pydantic import BaseModel, ValidationError

from src.cli.ui import Prompter

PORTFOLIO_TIMEOUT = 120.0
EXPIRED_HINT = "Verification codes expire after a few minutes and work only once."


class LoginResult(StrEnum):
    ALREADY = "already"
    DONE = "done"
    SKIPPED = "skipped"


class _Status(BaseModel):
    authenticated: bool


class _Started(BaseModel):
    auth_url: str


class _PortfolioRun(BaseModel):
    accounts_processed: int
    positions_stored: int


class _Error(BaseModel):
    detail: str


def _detail(response: httpx.Response) -> str:
    try:
        return _Error.model_validate_json(response.content).detail
    except ValidationError:
        return f"HTTP {response.status_code} {response.reason_phrase}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def is_authenticated(client: httpx.Client) -> bool:
    response = client.get("/auth/status")
    response.raise_for_status()
    return _Status.model_validate_json(response.content).authenticated


def login(client: httpx.Client, io: Prompter, open_url: Callable[[str], bool]) -> LoginResult:
    """Walk through E-Trade's OAuth, starting over whenever a step is refused."""
    if is_authenticated(client):
        io.info("E-Trade: already logged in.")
        return LoginResult.ALREADY

    while True:
        started = client.post("/auth/start")
        if started.status_code != 200:
            io.info(f"E-Trade would not start a login: {_detail(started)}")
            io.info("`./start doctor` checks your E-Trade keys.")
            if io.confirm("Try again?", default=True):
                continue
            return _skipped(io)

        url = _Started.model_validate_json(started.content).auth_url
        io.info("Log in to E-Trade in your browser and click Accept; E-Trade then shows a verification code.")
        io.info(url)
        if not open_url(url):
            io.info("Could not open a browser; open the link above yourself.")

        code = io.text("Verification code (leave blank to skip and run without E-Trade data)").strip()
        if not code:
            return _skipped(io)

        completed = client.post("/auth/complete", json={"verifier": code})
        if completed.status_code == 200:
            break
        io.info(f"E-Trade did not accept that code ({_detail(completed)}). {EXPIRED_HINT}")
        io.info("Starting over with a fresh link.")

    io.info("Logged in to E-Trade. Fetching your portfolio…")
    run = client.post("/pipeline/portfolio/run", timeout=PORTFOLIO_TIMEOUT)
    if run.status_code == 200:
        result = _PortfolioRun.model_validate_json(run.content)
        io.info(
            f"{_plural(result.positions_stored, 'position')} across "
            f"{_plural(result.accounts_processed, 'account')}."
        )
    else:
        io.info(f"Portfolio refresh failed ({_detail(run)}); it retries on its schedule.")
    return LoginResult.DONE


def _skipped(io: Prompter) -> LoginResult:
    io.info("Running without E-Trade data. Log in later with `./start login`.")
    return LoginResult.SKIPPED
