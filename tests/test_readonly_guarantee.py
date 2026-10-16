"""The E-Trade integration cannot place, preview, change or cancel orders.

E-Trade's OAuth 1.0a tokens are not scoped: the token this app holds could
trade. These tests hold the line that keeps it from doing so (see
src/etrade/__init__.py), statically over the source and at runtime against
the request guard. No network, no token.
"""
import importlib
import re
from pathlib import Path

import pytest
import requests
from requests.adapters import HTTPAdapter

import src.etrade
from src.etrade import ETradeAccessManager, ETradeAccounts, ETradeMarket
from src.etrade.client import BLOCKED_PATH_PATTERNS, ReadOnlyAdapter, check_request, order_endpoints_blocked, read_only
from src.etrade.errors import ReadOnlyViolationError

SRC = Path(__file__).resolve().parents[1] / "src"
ETRADE = SRC / "etrade"
# Split so this file never trips a search for the name itself.
ORDER_CLIENT = "ETrade" + "Order"


def _lines(path: Path) -> list[tuple[int, str]]:
    return list(enumerate(path.read_text(encoding="utf-8").splitlines(), start=1))


def _offences(pattern: re.Pattern[str], files: list[Path]) -> list[str]:
    return [f"{f.relative_to(SRC.parent)}:{n}: {line.strip()}" for f in files for n, line in _lines(f) if pattern.search(line)]


# ---------- static ----------


def test_nothing_in_src_names_the_order_client():
    # The test that catches a future change wiring order support in quietly.
    offences = _offences(re.compile(ORDER_CLIENT), sorted(SRC.rglob("*.py")))
    assert not offences, "the order client is referenced:\n" + "\n".join(offences)


def test_pyetrade_is_imported_only_by_the_etrade_package():
    # `import pyetrade` anywhere else would expose its order client as an attribute.
    importing = re.compile(r"^\s*(import pyetrade|from pyetrade\b)")
    files = [f for f in sorted(SRC.rglob("*.py")) if f != ETRADE / "__init__.py"]
    offences = _offences(importing, files)
    assert not offences, "pyetrade imported outside src/etrade/__init__.py:\n" + "\n".join(offences)


def test_no_etrade_api_url_is_written_outside_the_etrade_package():
    hosts = re.compile(r"\b(api|apisb)\.etrade\.com\b")
    files = [f for f in sorted(SRC.rglob("*.py")) if ETRADE not in f.parents]
    offences = _offences(hosts, files)
    assert not offences, "E-Trade API URL outside src/etrade/:\n" + "\n".join(offences)


def test_the_package_refuses_to_import_with_the_order_client_bound(monkeypatch):
    order_module = importlib.import_module("pyetrade.order")
    order_client = getattr(order_module, ORDER_CLIENT)
    # reload() re-runs the module in its existing namespace, as if this name
    # had been bound by an import there.
    monkeypatch.setattr(src.etrade, "sneaked_in", order_client, raising=False)

    with pytest.raises(ImportError, match="must not bind pyetrade's order client"):
        importlib.reload(src.etrade)

    monkeypatch.delattr(src.etrade, "sneaked_in")
    importlib.reload(src.etrade)  # clean again for the tests after this one


# ---------- runtime ----------

BASE = "https://api.etrade.com/v1/accounts/key123"


@pytest.mark.parametrize(
    "path",
    [
        "/orders/place",
        "/orders/preview",
        "/orders/cancel",
        "/orders/change/preview",
        "/orders/change/place",
        "/orders/place.json",
        "/Orders/Place",
    ],
)
def test_every_order_write_path_is_refused(path):
    with pytest.raises(ReadOnlyViolationError) as exc:
        check_request(BASE + path)
    assert exc.value.path.endswith(path)


@pytest.mark.parametrize(
    "path",
    ["/orders", "/orders.json", "/portfolio", "/balance", "/transactions", ""],
)
def test_reading_is_allowed_including_order_history(path):
    check_request(BASE + path)  # does not raise


def test_a_refusal_is_logged_at_critical():
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(m.record["level"].name), level="CRITICAL")
    try:
        with pytest.raises(ReadOnlyViolationError):
            check_request(BASE + "/orders/place")
    finally:
        logger.remove(sink)
    assert lines == ["CRITICAL"]


@pytest.fixture
def transport(monkeypatch):
    """What reached the network layer underneath the guard."""
    sent: list[tuple[str, str]] = []

    def fake_send(self, request, **kwargs):
        sent.append((request.method, request.url))
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        return response

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    return sent


@pytest.mark.parametrize("client_class", [ETradeAccounts, ETradeMarket])
def test_a_guarded_client_cannot_send_an_order_write(client_class, transport):
    client = read_only(client_class("ck", "cs", "token", "secret", dev=False))

    with pytest.raises(ReadOnlyViolationError):
        client.session.post(BASE + "/orders/place", json={})
    assert transport == []  # stopped before the network


def test_a_guarded_client_still_reads_order_history(transport):
    client = read_only(ETradeAccounts("ck", "cs", "token", "secret", dev=False))

    client.session.get(BASE + "/orders")

    assert transport == [("GET", BASE + "/orders")]


def test_the_token_renewer_is_guarded_too(transport):
    manager = read_only(ETradeAccessManager("ck", "cs", "token", "secret"))
    assert all(isinstance(a, ReadOnlyAdapter) for a in manager.session.adapters.values())


async def test_the_auth_layer_hands_out_only_guarded_clients(monkeypatch):
    from datetime import datetime, timezone

    from src.etrade.auth import ETradeAuth, ETradeTokens

    auth = ETradeAuth()
    auth._tokens = ETradeTokens("token", "secret", datetime.now(timezone.utc))

    for session in (await auth.get_market_session(), await auth.get_accounts_session()):
        assert {type(a) for a in session.session.adapters.values()} == {ReadOnlyAdapter}


def test_the_health_self_check_sees_every_pattern_blocked():
    assert order_endpoints_blocked()
    assert len(BLOCKED_PATH_PATTERNS) == 4
