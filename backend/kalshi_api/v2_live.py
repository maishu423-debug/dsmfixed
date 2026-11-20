"""
v2 live pipeline: model-script results + live context -> p_model and p_blend per Kalshi bucket.

Fitted parameters come from backend/config/fitted_v2.json, produced by eval/scripts/v2_fit_production.py with the
same walk-forward code as the evaluation, fitted on ALL history up to its build date (so it is the "next month"
fit). Without that file every fitted piece falls back to its data-free default (F2: lower bound, F4: none, F5:
w = 0, F6: a = 1, F7: the market mid), which is safe but not the evaluated model.
"""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from config import fixes

from . import daily_high as DH
from .algorithm import HORIZON_MIX, HORIZON_WEIGHTS, _in_bracket

ET = ZoneInfo("America/New_York")
ISSUE_TIMES = ["12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
FITTED_PATH = Path(__file__).resolve().parent.parent / "config" / "fitted_v2.json"
_fitted_cache = {}
_nbm_cache = {}


def load_fitted(path=FITTED_PATH):
    p = Path(path)
    if not p.exists():
        return None
    key = (str(p), p.stat().st_mtime)
    if key not in _fitted_cache:
        _fitted_cache.clear()
        _fitted_cache[key] = json.loads(p.read_text())
    return _fitted_cache[key]


def slot_for(now):
    loc = now.astimezone(ET)
    m = loc.hour * 60 + loc.minute
    best = min(ISSUE_TIMES, key=lambda h: abs(int(h[:2]) * 60 + int(h[3:]) - m))
    return best


def model_weights():
    mix = dict(HORIZON_MIX)
    if fixes.FIX_F5:
        mix["1h"] = 0.0
    return {m: mix[h] * w for h, pair in HORIZON_WEIGHTS.items() for m, w in pair.items() if mix[h] * w > 0}


def _nbm(now):
    day = now.astimezone(ET).date()
    hit = _nbm_cache.get(day)
    if hit and (now - hit[0]).total_seconds() < 600:
        return hit[1]
    try:
        from weather.nbm import fetch_latest_txn
        val, _ = fetch_latest_txn(day, now)
    except Exception:
        val = hit[1] if hit else None
    _nbm_cache[day] = (now, val)
    return val


def build_context(observations, now, known_bound):
    """
    Live context for daily_high.compose() from today's 5-min feed observations (dicts with observed_at, temp_f,
    weather) plus the NWS readings / DSM caches in kalshi_api.nws and the NBM guidance.
    """
    from weather.asos_parser import parse_utc
    from . import nws
    today = now.astimezone(ET).date()
    obs = []
    for o in observations or []:
        t = parse_utc(o.get("observed_at"))
        if t is None or o.get("temp_f") is None or t > now or t.astimezone(ET).date() != today:
            continue
        obs.append((t, int(round((float(o["temp_f"]) - 32) * 5 / 9)), o.get("weather") or ""))
    ctx = {"issue": slot_for(now), "known_bound": known_bound}
    if obs:
        cmax = max(c for _, c, _ in obs)
        at = [t for t, c, _ in obs if c == cmax]
        ctx.update(hf_c_max=float(cmax), n_at_max=len(at), max_times=[x.isoformat() for x in at],
                   hours_since_max=(obs[-1][0] - max(at)).total_seconds() / 3600,
                   drop_from_max_c=float(cmax - obs[-1][1]))
        six = datetime(today.year, today.month, today.day, 6, tzinfo=ET)
        ctx["conv_so_far"] = any(any(k in wx for k in ("TS", "RA", "SH")) for t, _, wx in obs if t >= six)
    try:
        r = [f for _, f in nws._readings_today(now) if f is not None]
        ctx["met_exact_max"] = max(r) if r else None
        dsm, _ = nws._todays_dsm_high(now)
        ctx["dsm_high"] = dsm.get("high_f") if dsm else None
        if dsm and dsm.get("report_time_lst"):           # F11: the DSM covers the climate day up to its report time
            from datetime import date as _date
            d = _date.fromisoformat(dsm["date"])
            h, m = map(int, dsm["report_time_lst"].split(":"))
            lst = timezone(timedelta(hours=-5))
            ctx["dsm_report_utc"] = (datetime(d.year, d.month, d.day, tzinfo=lst)
                                     + timedelta(hours=h, minutes=m)).astimezone(timezone.utc).isoformat()
        data = nws._cached("nws_readings", nws.NWS_READINGS_TTL_SEC, nws._load_nws_readings)
        ctx["six_groups"] = [[avail.isoformat(), c] for _, _, c, avail in (data.get("max6h") or []) if avail <= now]
    except Exception:
        pass
    ctx["nbm"] = _nbm(now) if fixes.FIX_F5 else None
    return ctx


def bucket_probs(results, buckets, ctx, fitted=None):
    """
    results: model-script outputs (with fut_max_hist); buckets: [{"ticker", "floor", "cap", "yes_bid", "yes_ask",
    "live"}]. Returns (p_model list, p_blend list or None, info).
    """
    fitted = fitted if fitted is not None else (load_fitted() or {})
    hists = {k: v["fut_max_hist"] for k, v in results.items()
             if isinstance(v, dict) and "error" not in v and v.get("fut_max_hist")}
    if not hists:
        return None, None, {"error": "no fut_max_hist in the model results (FIX_F1 off?)"}
    issue = ctx.get("issue", "14:00")
    anchor = None
    if fixes.FIX_F5 and fitted.get("w", {}).get(issue, 0) > 0:
        anchor = {"w": fitted["w"][issue], "bias": fitted.get("bias", 0.0)}
    pmf, info = DH.compose(ctx, hists, model_weights(),
                           f2_params=fitted.get("f2") if fixes.FIX_F2 else None,
                           f4_params=fitted.get("f4") if fixes.FIX_F4 else None, anchor=anchor,
                           f11=fixes.FIX_F11)
    info["pmf"] = {int(k): float(v) for k, v in pmf.items()}
    p = np.array([sum(v for t, v in pmf.items() if _in_bracket(t, b.get("floor"), b.get("cap"))) for b in buckets])
    if fixes.FIX_F6:
        from .recalibration import recalibrate
        p = recalibrate(p, fitted.get("a", {}).get(issue, 1.0))
    p_blend = None
    if fixes.FIX_F7:
        from .market_blend import blend
        mids = np.array([((b.get("yes_bid") or 0.0) + (b.get("yes_ask") if b.get("yes_ask") is not None else 1.0)) / 2
                         for b in buckets])
        p_mid = mids / mids.sum() if mids.sum() > 0 else mids
        p_blend = blend(p, p_mid, fitted.get("f7", [0.0, 1.0, 0.0]), live=[b.get("live", True) for b in buckets])
    return p.tolist(), (p_blend.tolist() if p_blend is not None else None), info
