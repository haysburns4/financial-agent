"""OAuth handshake failures surface as a domain error, not vendor HTML."""
import pytest
from requests_oauthlib.oauth1_session import TokenRequestDenied

from src.etrade.auth import ETradeAuth, ETradeAuthError

# What E-Trade actually returns when it does not recognise the consumer key.
_REJECTED = (
    "Token request failed with code 401, response was '<!doctype html>"
    "<p><b>Message</b> oauth_problem=consumer_key_rejected</p></body></html>'."
)


class _DeniedOAuth:
    """Stands in for pyetrade.ETradeOAuth, refusing at every step."""

    def __init__(self, consumer_key: str, consumer_secret: str) -> None:
        self.consumer_key = consumer_key

    def get_request_token(self) -> str:
        raise TokenRequestDenied(_REJECTED, response=None)

    def get_access_token(self, verifier: str) -> dict:
        raise TokenRequestDenied(_REJECTED, response=None)


def test_start_auth_translates_a_rejected_consumer_key():
    with pytest.raises(ETradeAuthError) as exc:
        ETradeAuth(oauth_factory=_DeniedOAuth).start_auth()

    assert exc.value.oauth_problem == "consumer_key_rejected"
    assert "request token" in str(exc.value)


async def test_complete_auth_translates_a_rejected_verifier():
    auth = ETradeAuth(oauth_factory=_DeniedOAuth)
    with pytest.raises(ETradeAuthError):
        auth.start_auth()  # fails, but leaves the session in place as a real call would

    with pytest.raises(ETradeAuthError) as exc:
        await auth.complete_auth("verifier")

    assert exc.value.oauth_problem == "consumer_key_rejected"
    assert "access token" in str(exc.value)


def test_unparseable_denial_falls_back_to_the_raw_message():
    class _WeirdlyDenied(_DeniedOAuth):
        def get_request_token(self) -> str:
            raise TokenRequestDenied("Token request failed with code 503", response=None)

    with pytest.raises(ETradeAuthError) as exc:
        ETradeAuth(oauth_factory=_WeirdlyDenied).start_auth()

    # No oauth_problem to parse, and reading exc.status_code would blow up on a
    # denial carrying no response — fall back to the raw message.
    assert exc.value.oauth_problem is None
    assert "code 503" in str(exc.value)
