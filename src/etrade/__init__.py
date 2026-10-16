"""E-Trade integration — read-only by construction.

This app reads accounts, positions, balances and quotes. It does not place,
preview, change or cancel an order:

- this module is the only place pyetrade is imported, and it binds only the
  auth, accounts and market clients — not pyetrade's order client;
- every client that carries an access token sends its requests through
  `src.etrade.client.ReadOnlyAdapter`, which refuses order-write paths
  before they leave the process;
- tests/test_readonly_guarantee.py fails if either layer is bypassed.
"""
import inspect

from pyetrade import ETradeAccessManager, ETradeAccounts, ETradeMarket, ETradeOAuth

__all__ = ["ETradeAccessManager", "ETradeAccounts", "ETradeMarket", "ETradeOAuth"]

# pyetrade's order module; nothing bound in this package may come from it.
_ORDER_MODULE = "pyetrade.order"


def _from_order_module(value: object) -> bool:
    # getmodule resolves a class or function to its module, and a module to itself.
    module = inspect.getmodule(value)
    return module is not None and module.__name__ == _ORDER_MODULE


_bound_from_orders = sorted(name for name, value in list(globals().items()) if _from_order_module(value))
if _bound_from_orders:
    raise ImportError(f"src.etrade must not bind pyetrade's order client; found {_bound_from_orders}")
