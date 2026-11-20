"""
PART 0 safety: prove that no order call can reach Kalshi from this copy.

1. Runtime: every order entry point raises LiveTradingDisabledError, and the network layer (urllib urlopen and
   requests) is never touched - it is patched to fail the test if called.
2. Static: scan every .py file of the project; any code that could issue a non-GET request to Kalshi must be inside
   a function that calls require_live_trading(). No file imports the third-party `kalshi` SDK.
"""
import ast
import re
from pathlib import Path

import pytest

from kalshi_api import client, live, trading_guard
from kalshi_api.trading_guard import LiveTradingDisabledError

ROOT = Path(__file__).resolve().parents[2]          # project root (model2v2dsmbugfixed)


@pytest.fixture
def no_network(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append((a, k))
        raise AssertionError("network called while live trading is disabled")

    import urllib.request
    import requests
    monkeypatch.setattr(client, "urlopen", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(requests.Session, "request", boom)
    # credentials present -> the only thing stopping the order is the guard
    monkeypatch.setattr(client, "_auth_headers_full", lambda m, p: {"KALSHI-ACCESS-KEY": "x"})
    return calls


def test_flag_is_false():
    assert trading_guard.LIVE_TRADING_ENABLED is False


ORDER_CALLS = [
    lambda: client.post_json("/portfolio/orders", {"ticker": "X", "side": "yes", "count": 1}),
    lambda: client.create_order({"ticker": "X", "side": "yes", "count": 1}),
    lambda: client.batch_create_orders([{"ticker": "X"}]),
    lambda: client.cancel_order("abc"),
    lambda: client.batch_cancel_orders(["abc"]),
    lambda: client.amend_order("abc", {"yes_price": 50}),
    lambda: client.decrease_order("abc", 1),
    lambda: client._send_write("DELETE", "/portfolio/orders/abc"),
    lambda: live.place_live_sell("X", "yes", 50, 1),
    lambda: live.place_live_order("X", "yes", 50, 1),
    lambda: live.cancel_live_order("abc"),
    lambda: live.get_state(),
]


@pytest.mark.parametrize("call", ORDER_CALLS)
def test_every_order_path_raises_without_network(call, no_network):
    with pytest.raises(LiveTradingDisabledError):
        call()
    assert no_network == []


def test_profit_watcher_sell_is_blocked(no_network, monkeypatch):
    """A profit-target hit goes through live.place_live_sell, which must raise (logged, nothing sent)."""
    from kalshi_api import profit_watcher as pw
    seen = []
    monkeypatch.setattr(pw.logger, "error", lambda msg: seen.append(msg))
    pw._positions.clear()
    pw._positions["KXHIGHMIA-T"] = {"side": "yes", "avg_price_cents": 10, "contracts": 5}
    monkeypatch.setattr(pw, "_enabled", True)
    pw._check_and_sell("KXHIGHMIA-T", 90)            # 80c profit: would sell
    assert any("LIVE_TRADING_ENABLED is False" in m for m in seen)
    assert no_network == []
    assert "KXHIGHMIA-T" not in pw._pending_sells


# ── static scan ──────────────────────────────────────────────────────────────

_SKIP_DIRS = {"venv", ".venv", "node_modules", ".next", "__pycache__", ".git", "raw"}
_WRITE_PAT = re.compile(r"""method\s*=\s*["'](POST|DELETE|PUT|PATCH)["']|requests\.(post|delete|put|patch)\(|"""
                        r"""\.(post|delete|put|patch)\(\s*f?["'][^"']*kalshi""", re.I)


def _py_files():
    for p in ROOT.rglob("*.py"):
        if _SKIP_DIRS.intersection(p.relative_to(ROOT).parts):
            continue
        yield p


def _guarded_functions(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = ast.dump(node)
            if "require_live_trading" in src:
                names.add((node.lineno, node.end_lineno))
    return names


def test_static_no_unguarded_kalshi_write():
    offenders = []
    for p in _py_files():
        text = p.read_text(encoding="utf-8", errors="replace")
        if not _WRITE_PAT.search(text):
            continue
        tree = ast.parse(text)
        spans = _guarded_functions(tree)
        for i, line in enumerate(text.splitlines(), 1):
            if not _WRITE_PAT.search(line):
                continue
            # writes to non-Kalshi services (Telegram / email notifier) are allowed
            if p.name in ("notifier.py",) or p.name.startswith("test_"):
                continue
            if not any(a <= i <= b for a, b in spans):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert offenders == [], "unguarded write request(s):\n" + "\n".join(offenders)


def test_static_no_order_endpoint_outside_guarded_code():
    offenders = []
    for p in _py_files():
        if p.name.startswith("test_"):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "portfolio/orders" not in text:
            continue
        spans = _guarded_functions(ast.parse(text))
        for i, line in enumerate(text.splitlines(), 1):
            if "portfolio/orders" in line and not line.strip().startswith("#") and not any(a <= i <= b for a, b in spans):
                offenders.append(f"{p.relative_to(ROOT)}:{i}")
    assert offenders == [], offenders


def test_static_no_kalshi_sdk_import():
    pat = re.compile(r"^\s*(import kalshi\b(?!_)|from kalshi\s+import|from kalshi\.)", re.M)
    hits = [str(p.relative_to(ROOT)) for p in _py_files() if pat.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert hits == []
