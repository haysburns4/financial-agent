"""E-Trade OAuth errors, importable without src.config (setup tooling uses them)."""
import re

from requests_oauthlib.oauth1_session import TokenRequestDenied


class ETradeAuthError(Exception):
    """E-Trade refused the OAuth handshake.

    Raised instead of leaking `TokenRequestDenied` and a page of HTML;
    `oauth_problem` carries E-Trade's reason, e.g. `consumer_key_rejected`.
    """

    def __init__(self, message: str, oauth_problem: str | None = None) -> None:
        super().__init__(message)
        self.oauth_problem = oauth_problem


def denied(exc: TokenRequestDenied, stage: str) -> ETradeAuthError:
    # Don't touch exc.status_code — it dereferences exc.response, which is
    # optional. The reason we want is in the body E-Trade echoed back.
    match = re.search(r"oauth_problem=([A-Za-z_]+)", str(exc))
    problem = match.group(1) if match else None
    return ETradeAuthError(
        f"E-Trade rejected the {stage} request: {problem or str(exc)[:200]}",
        problem,
    )
