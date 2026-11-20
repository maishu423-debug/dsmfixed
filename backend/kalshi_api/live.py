"""
Live-order module stub for the research copy.

profit_watcher.py imports `get_state` and `place_live_sell` from here. The original tree did not contain this
module (the import would have failed). In this copy every function that would trade raises
LiveTradingDisabledError via kalshi_api.trading_guard; get_state() also refuses because it only exists to drive
live selling.
"""
from .trading_guard import require_live_trading


def get_state():
    require_live_trading("live.get_state")


def place_live_order(*args, **kwargs):
    require_live_trading("live.place_live_order")


def place_live_sell(*args, **kwargs):
    require_live_trading("live.place_live_sell")


def cancel_live_order(*args, **kwargs):
    require_live_trading("live.cancel_live_order")
