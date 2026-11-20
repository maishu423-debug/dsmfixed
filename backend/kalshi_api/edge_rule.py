"""
F10: the trading rule (backtest and paper mode only - real orders stay blocked by kalshi_api/trading_guard.py).

Replaces the v1 rule ("YES on the top bucket, NO on every other bucket, at any ask <= 98c"), which lost about -$46k
in the v1 holdout replay.

Taker mode (default): for each live bucket and each side,
    edge = p_side - ask_side                       p_side = p_blend (YES) or 1 - p_blend (NO)
    fee  = ceil-to-cent(0.07 * P * (1 - P))         Kalshi taker fee per contract at price P
    trade if edge > fee + buffer                     buffer = F10_BUFFER_CENTS (default 3c)
    and, if the side costs more than 90c, only when edge - fee > 5c
Maker mode (optional): post a limit at fair value minus the buffer, L = floor_cent(p_side - buffer), only if L is
below the ask (else it would cross) and p_side - L exceeds the maker fee; the backtest counts a fill only if a later
trade printed at or through L (optimistic: ignores adverse selection and queue position).
Always: at most F10_MAX_ENTRIES_PER_MARKET_DAY entries per market per day; size capped by the contracts traded in that
market over the previous F10_VOLUME_LOOKBACK_MIN minutes.
"""
import math

from config import fixes

TAKER_RATE = 0.07
MAKER_RATE = 0.0175


def fee_per_contract(price, rate=TAKER_RATE):
    """Kalshi fee per contract in dollars, rounded UP to the cent (price in dollars)."""
    return math.ceil(round(rate * price * (1 - price) * 100, 6)) / 100.0


def decide(buckets, *, buffer_cents=None, mode="taker", entered=(), stake_dollars=100.0):
    """
    buckets: [{"ticker", "label", "p_blend", "yes_ask", "yes_bid", "recent_volume", "live"}], prices in dollars.
    entered: tickers already entered today (one entry per market per day).
    Returns legs [{"ticker", "label", "side", "price", "edge", "fee", "contracts", "mode"}].
    """
    buf = (fixes.F10_BUFFER_CENTS if buffer_cents is None else buffer_cents) / 100.0
    hi_px = fixes.F10_HIGH_PRICE_CENTS / 100.0
    hi_edge = fixes.F10_HIGH_PRICE_MIN_EDGE_CENTS / 100.0
    legs = []
    taken = set(entered)
    for b in buckets:
        if not b.get("live", True) or b.get("p_blend") is None or b["ticker"] in taken:
            continue
        ya, yb = b.get("yes_ask"), b.get("yes_bid")
        if ya is None:
            continue
        no_ask = 1.0 - (yb if yb is not None else 0.0)
        best = None
        for side, p, ask in (("yes", b["p_blend"], ya), ("no", 1.0 - b["p_blend"], no_ask)):
            if mode == "taker":
                if not (0 < ask < 1):
                    continue
                fee = fee_per_contract(ask)
                edge = p - ask
                if edge <= fee + buf or (ask > hi_px and edge - fee <= hi_edge):
                    continue
                price = ask
            else:
                price = math.floor((p - buf) * 100 + 1e-9) / 100.0
                if price < 0.01 or price >= ask:
                    continue
                fee = fee_per_contract(price, MAKER_RATE)
                edge = p - price
                if edge <= fee or (price > hi_px and edge - fee <= hi_edge):
                    continue
            score = round(edge - fee, 6)                     # ties (common in maker mode) go to the first side, YES
            if best is None or score > best["_score"]:
                best = {"ticker": b["ticker"], "label": b.get("label", ""), "side": side, "price": price,
                        "edge": round(edge, 4), "fee": fee, "mode": mode, "_score": score}
        if best is None:
            continue
        cap = int(math.floor(stake_dollars / best["price"]))
        vol = b.get("recent_volume")
        if vol is not None:
            cap = min(cap, int(vol))
        if cap < 1:
            continue
        best["contracts"] = cap
        best.pop("_score")
        legs.append(best)
        taken.add(b["ticker"])
    return legs
