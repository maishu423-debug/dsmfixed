"""
F9: strike metadata for KXHIGHMIA markets.

Some markets come back from Kalshi without floor_strike / cap_strike (25 settled markets in Jan-Feb 2025). With both
missing, algorithm._in_bracket(t, None, None) returned True: every temperature counted as YES for that bucket.

resolve_strikes() fills them in, in this order:
  1. the market's own floor_strike / cap_strike when present;
  2. the ticker suffix:  -B<x>  = "between":  floor x - 0.5, cap x + 0.5   (B73.5 = 73-74 F)
                         -T<x>  = an open-ended tail; the side is taken from the market's rules / title text
                                  ("greater than x" -> greater, floor x; "less than x" -> less, cap x) or, failing
                                  that, from the sibling markets of the same event (above every "B" bucket -> greater,
                                  below every one -> less);
  3. otherwise the market is UNTRADEABLE: it is skipped, never treated as YES.
"""
import re

_TICKER = re.compile(r"-([TB])(\d+(?:\.\d+)?)$")
_GREATER = re.compile(r"\b(greater than|above|or above|or higher|>)\b", re.I)
_LESS = re.compile(r"\b(less than|below|or below|or lower|<)\b", re.I)

UNTRADEABLE = "untradeable"


def _f(x):
    try:
        return None if x is None or x != x else float(x)
    except (TypeError, ValueError):
        return None


def parse_ticker(ticker, siblings=(), text=None):
    """(floor, cap, strike_type) from the ticker (see module docstring), or None if it cannot be determined."""
    m = _TICKER.search(ticker or "")
    if not m:
        return None
    kind, val = m.group(1), float(m.group(2))
    if kind == "B":
        return val - 0.5, val + 0.5, "between"
    if text:
        g, l_ = bool(_GREATER.search(text)), bool(_LESS.search(text))
        if g and not l_:
            return val, None, "greater"
        if l_ and not g:
            return None, val, "less"
    bvals = []
    for s in siblings or ():
        sm = _TICKER.search(s or "")
        if sm and sm.group(1) == "B":
            bvals.append(float(sm.group(2)))
    if bvals:
        if val > max(bvals):
            return val, None, "greater"
        if val < min(bvals):
            return None, val, "less"
    return None


def resolve_strikes(market, siblings=()):
    """
    {"floor", "cap", "strike_type", "tradeable", "strike_source"} for one market dict (Kalshi API or normalized).
    `siblings` are the tickers of the other markets of the same event.
    """
    floor = _f(market.get("floor_strike", market.get("floor")))
    cap = _f(market.get("cap_strike", market.get("cap")))
    if floor is not None or cap is not None:
        st = market.get("strike_type") or ("between" if floor is not None and cap is not None
                                           else "greater" if floor is not None else "less")
        return {"floor": floor, "cap": cap, "strike_type": st, "tradeable": True, "strike_source": "metadata"}
    text = " ".join(str(market.get(k) or "") for k in ("rules_primary", "rules", "title", "subtitle", "yes_sub_title"))
    parsed = parse_ticker(market.get("ticker"), siblings, text.strip() or None)
    if parsed is None:
        return {"floor": None, "cap": None, "strike_type": UNTRADEABLE, "tradeable": False,
                "strike_source": "unparseable"}
    f, c, st = parsed
    return {"floor": f, "cap": c, "strike_type": st, "tradeable": True, "strike_source": "ticker"}
