"""F10: edge-after-fees rule (paper/backtest only); the v1 'YES top / NO all' rule is disabled."""
import pytest

from config import fixes
from kalshi_api import algorithm as alg
from kalshi_api import edge_rule as ER


def B(ticker, p, ask, bid, vol=None, live=True):
    return {"ticker": ticker, "label": ticker, "p_blend": p, "yes_ask": ask, "yes_bid": bid, "recent_volume": vol,
            "live": live}


def test_fee_is_ceil_to_cent():
    assert ER.fee_per_contract(0.50) == 0.02          # 0.07*0.25 = 1.75c -> 2c
    assert ER.fee_per_contract(0.93) == 0.01          # 0.46c -> 1c
    assert ER.fee_per_contract(0.10) == 0.01          # 0.63c -> 1c


def test_taker_trades_only_above_fee_plus_buffer():
    legs = ER.decide([B("A", 0.60, 0.50, 0.48)], buffer_cents=3)
    assert len(legs) == 1 and legs[0]["side"] == "yes" and legs[0]["price"] == 0.50
    assert ER.decide([B("A", 0.54, 0.50, 0.48)], buffer_cents=3) == []          # 4c edge < 2c fee + 3c
    legs = ER.decide([B("A", 0.30, 0.50, 0.45)], buffer_cents=3)                 # NO at 55c, p_no 0.70
    assert legs[0]["side"] == "no" and legs[0]["price"] == pytest.approx(0.55)


def test_expensive_side_needs_5c_after_fees():
    # NO costs 93c (yes_bid 7c); p_no 0.97: edge 4c - fee 1c = 3c <= 5c -> no trade
    assert ER.decide([B("A", 0.03, 0.10, 0.07)], buffer_cents=1) == []
    legs = ER.decide([B("A", 0.00, 0.10, 0.07)], buffer_cents=1)                 # edge 7c - 1c = 6c > 5c
    assert legs and legs[0]["side"] == "no"


def test_one_entry_per_market_and_volume_cap():
    assert ER.decide([B("A", 0.60, 0.50, 0.48)], entered={"A"}) == []
    legs = ER.decide([B("A", 0.60, 0.50, 0.48, vol=7)], stake_dollars=100)
    assert legs[0]["contracts"] == 7
    assert ER.decide([B("A", 0.60, 0.50, 0.48, vol=0)]) == []
    assert ER.decide([B("A", 0.60, 0.50, 0.48, live=False)]) == []


def test_maker_posts_below_fair_value():
    legs = ER.decide([B("A", 0.60, 0.62, 0.50)], buffer_cents=3, mode="maker")
    assert legs[0]["side"] == "yes" and legs[0]["price"] == 0.57 and legs[0]["mode"] == "maker"
    # the YES limit (57c) would cross a 55c ask: only the NO side can be posted
    legs = ER.decide([B("A", 0.60, 0.55, 0.50)], buffer_cents=3, mode="maker")
    assert legs[0]["side"] == "no" and legs[0]["price"] == 0.37


def test_old_rule_disabled_under_f10(monkeypatch):
    called = {}
    monkeypatch.setattr(alg, "_f10_recommendation", lambda a, r: called.setdefault("f10", {"status": "waiting",
                                                                                         "trades": []}))
    with fixes.override(FIX_F10=True):
        rec = alg.get_recommendation([], results={})
    assert called and rec["trades"] == []


def test_f10_recommendation_uses_blend(monkeypatch):
    from kalshi_api import forecast, v2_live
    monkeypatch.setattr(forecast, "get_forecast_cache", lambda: {"asos_current": {"v2_ctx": {"hf_c_max": 30.0}}})
    monkeypatch.setattr(v2_live, "bucket_probs", lambda res, b, ctx, fitted=None: ([0.2, 0.8], [0.1, 0.9], {"mean": 86}))
    monkeypatch.setattr(alg, "_entered_today", lambda: set())
    analyses = [{"ticker": "T1", "label": "84-85", "floor": 84, "cap": 85, "yes_ask_cents": 30, "yes_bid_cents": 28,
                 "is_dead": False},
                {"ticker": "T2", "label": "86-87", "floor": 86, "cap": 87, "yes_ask_cents": 70, "yes_bid_cents": 68,
                 "is_dead": False}]
    rec = alg._f10_recommendation(analyses, {})
    sides = {(t["ticker"], t["side"]) for t in rec["trades"]}
    assert sides == {("T1", "no"), ("T2", "yes")}          # NO T1 at 72c (p 0.9), YES T2 at 70c (p 0.9)
