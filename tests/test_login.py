"""Guided E-Trade login against a fake API (httpx.MockTransport)."""
import json
from collections import deque

import httpx

from src.cli.login import LoginResult, login
from tests.cli_fakes import ScriptedPrompter

AUTH_URL = "https://us.etrade.com/e/t/etws/authorize?key=k&token=t"
EXPIRED = "E-Trade rejected the access token request: token_rejected"


class FakeApi:
    """Just the auth and portfolio endpoints, scripted per call."""

    def __init__(
        self,
        authenticated: bool = False,
        start: list[int] | None = None,
        complete: list[int] | None = None,
    ) -> None:
        self.authenticated = authenticated
        self.start = deque(start or [])
        self.complete = deque(complete or [])
        self.calls: list[str] = []
        self.verifiers: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(f"{request.method} {request.url.path}")
        match request.url.path:
            case "/auth/status":
                return httpx.Response(200, json={"authenticated": self.authenticated, "session_age_minutes": None})
            case "/auth/start":
                status = self.start.popleft() if self.start else 200
                if status != 200:
                    return httpx.Response(status, json={"detail": "E-Trade rejected the request token request: consumer_key_rejected"})
                return httpx.Response(200, json={"auth_url": AUTH_URL})
            case "/auth/complete":
                self.verifiers.append(json.loads(request.content)["verifier"])
                status = self.complete.popleft() if self.complete else 200
                if status != 200:
                    return httpx.Response(status, json={"detail": EXPIRED})
                self.authenticated = True
                return httpx.Response(200, json={"authenticated": True})
            case "/pipeline/portfolio/run":
                return httpx.Response(200, json={"status": "success", "accounts_processed": 2, "positions_stored": 3})
        return httpx.Response(404)


def _client(api: FakeApi) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(api), base_url="http://api.test")


class Browser:
    def __init__(self, works: bool = True) -> None:
        self.works = works
        self.opened: list[str] = []

    def __call__(self, url: str) -> bool:
        self.opened.append(url)
        return self.works


def test_already_logged_in_asks_nothing():
    api = FakeApi(authenticated=True)
    io = ScriptedPrompter()
    assert login(_client(api), io, Browser()) is LoginResult.ALREADY
    assert api.calls == ["GET /auth/status"]


def test_successful_login_refreshes_the_portfolio():
    api = FakeApi()
    io = ScriptedPrompter([("Verification code", "ABC12")])
    browser = Browser()

    assert login(_client(api), io, browser) is LoginResult.DONE

    assert browser.opened == [AUTH_URL]
    assert AUTH_URL in io.output  # printed too, in case no browser opens
    assert api.calls[-1] == "POST /pipeline/portfolio/run"
    assert "3 positions across 2 accounts." in io.output


def test_rejected_code_starts_over_with_a_fresh_link():
    api = FakeApi(complete=[502])
    io = ScriptedPrompter([("Verification code", "OLD99"), ("Verification code", "NEW42")])
    browser = Browser()

    assert login(_client(api), io, browser) is LoginResult.DONE

    assert api.verifiers == ["OLD99", "NEW42"]
    assert api.calls.count("POST /auth/start") == 2
    assert len(browser.opened) == 2
    assert any("token_rejected" in line and "expire" in line for line in io.output)
    assert not io.script


def test_blank_code_skips_without_completing():
    api = FakeApi()
    io = ScriptedPrompter([("leave blank to skip", "")])

    assert login(_client(api), io, Browser()) is LoginResult.SKIPPED

    assert "POST /auth/complete" not in api.calls
    assert any("./start login" in line for line in io.output)


def test_refused_start_can_be_retried_or_skipped():
    api = FakeApi(start=[502, 502])
    io = ScriptedPrompter([("Try again", "y"), ("Try again", "n")])

    assert login(_client(api), io, Browser()) is LoginResult.SKIPPED

    assert api.calls.count("POST /auth/start") == 2
    assert any("consumer_key_rejected" in line for line in io.output)


def test_no_browser_still_gives_the_link():
    io = ScriptedPrompter([("Verification code", "ABC12")])
    login(_client(FakeApi()), io, Browser(works=False))
    assert AUTH_URL in io.output
    assert any("open the link above yourself" in line for line in io.output)
