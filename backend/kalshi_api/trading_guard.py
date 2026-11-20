"""
Hard kill-switch for every code path that could send an order to Kalshi.

This project (model2v2dsmbugfixed) is a research / paper-trading copy. It must NEVER place, cancel, amend or
batch real orders. Every function that could do so calls `require_live_trading(<action>)` first, which raises
LiveTradingDisabledError while LIVE_TRADING_ENABLED is False.

Guarded paths (see backend/tests/test_trading_guard.py, which proves none of them reaches the network):
  * kalshi_api.client.post_json            - the only signed write request in the codebase (all Kalshi POST
                                             endpoints are order endpoints: create, batch create, amend, decrease)
  * kalshi_api.client.create_order / cancel_order / amend_order / decrease_order / batch_create_orders /
    batch_cancel_orders                    - explicit entry points, all routed through the guard
  * kalshi_api.live.*                      - the live-order module that profit_watcher.py imports (it was absent
                                             from the original tree; this copy provides a stub that only raises)

Do not flip this flag in this folder. There is intentionally no environment-variable override.
"""

LIVE_TRADING_ENABLED = False


class LiveTradingDisabledError(RuntimeError):
    """Raised whenever a real-order code path is called while LIVE_TRADING_ENABLED is False."""


def require_live_trading(action: str) -> None:
    if not LIVE_TRADING_ENABLED:
        raise LiveTradingDisabledError(
            f"Blocked '{action}': LIVE_TRADING_ENABLED is False in kalshi_api/trading_guard.py "
            f"(this copy never sends real orders to Kalshi)."
        )
