# Read-only E-Trade

This app only reads your E-Trade accounts. That cannot be
left to the credentials. E-Trade's OAuth 1.0a has no permission scopes, so
there is no read-only token to ask for.

The code itself blocks trading at two layers:

- **Imports.** `src/etrade/__init__.py` is the only place `pyetrade` is
  imported, and it binds only the auth, accounts and market clients, never
  pyetrade's order client. The package refuses to import if anything from
  that order client is bound.
- **Requests.** Every E-Trade client that carries the access token sends
  through one transport, `src/etrade/client.py`, which refuses any request to
  an order place, preview, change or cancel path before it leaves the process.
  Reading order history stays allowed.

`tests/test_readonly_guarantee.py` enforces both, with no network or token.
It fails if anything under `src/` names the order client, imports `pyetrade`
anywhere but `src/etrade/__init__.py`, or writes an E-Trade API URL outside
`src/etrade/`, or if the auth layer hands out a client without the guard. `/health` reports
`"etrade": {"mode": "read_only", "order_endpoints_blocked": true}`, and the
startup log confirms the guard is active.
