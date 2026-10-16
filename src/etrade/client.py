"""The one place every E-Trade request passes through, and the read-only rule
it enforces.

pyetrade clients each build their own `requests` session. `read_only(client)`
mounts `ReadOnlyAdapter` on that session for every URL, so each request it
sends — redirects included — reaches `ReadOnlyAdapter.send`, which checks the
path before anything leaves the process. The auth layer (src/etrade/auth.py)
wraps every client that carries an access token this way.
"""
from typing import Protocol, TypeVar
from urllib.parse import urlsplit

from loguru import logger
from requests import PreparedRequest, Response, Session
from requests.adapters import HTTPAdapter

from src.etrade.errors import ReadOnlyViolationError

BLOCKED_PATH_PATTERNS = (
    "/orders/place",
    "/orders/preview",
    "/orders/cancel",
    "/orders/change",
)


def blocked_path(url: str) -> str | None:
    """`url`'s path if it is an order-write endpoint, else None."""
    path = urlsplit(url).path
    lowered = path.lower()
    return path if any(pattern in lowered for pattern in BLOCKED_PATH_PATTERNS) else None


def check_request(url: str) -> None:
    """Raise ReadOnlyViolationError if `url` is an order-write endpoint."""
    if (path := blocked_path(url)) is not None:
        logger.critical("E-Trade read-only guard blocked a request to {}; this should never happen", path)
        raise ReadOnlyViolationError(path)


class ReadOnlyAdapter(HTTPAdapter):
    """A transport that refuses order-write requests before sending them."""

    def send(self, request: PreparedRequest, **kwargs) -> Response:
        check_request(request.url or "")
        return super().send(request, **kwargs)


class _HasSession(Protocol):
    session: Session


Client = TypeVar("Client", bound=_HasSession)


def read_only(client: Client) -> Client:
    """`client` with its session sending through ReadOnlyAdapter."""
    adapter = ReadOnlyAdapter()
    for prefix in ("https://", "http://"):
        client.session.mount(prefix, adapter)
    return client


def order_endpoints_blocked() -> bool:
    """A self-check for /health: does the guard still reject every write path?"""
    return all(blocked_path(f"/v1/accounts/key{pattern}") is not None for pattern in BLOCKED_PATH_PATTERNS)
