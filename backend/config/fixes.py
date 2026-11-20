"""
Feature flags for the v2 bug fixes (see eval/REPORT_v1.md for the problems and eval/REPORT_v2.md for the results).

Every fix is behind its own flag, default True. With every flag False this copy reproduces the v1 forecasts
exactly (eval/tests/test_v1_reproduction.py), so each fix's effect can be measured and reverted on its own.

  FIX_F1   floor/start the simulated paths at the whole-F candidates of the whole-C reading, not at its rounded value
  FIX_F2   the known high so far is a distribution over its whole-F candidates (empirical P(upper))
  FIX_F3   METAR/SPECI T-group parsed exactly; RMK 6-hour max group (1snTTT) used from its availability time
  FIX_F4   settlement layer: distribution of (CLI high - live max) added after the path simulation
  FIX_F5   paths simulated to 19:00 local, anchored to NBM max-temp guidance; 1-hour models dropped
  FIX_F6   walk-forward recalibration of the bucket probabilities (temperature scaling per issue time)
  FIX_F7   second output p_blend = logistic blend of the Kalshi mid and the model
  FIX_F8   DSM_MAX_EXCESS_F guard: accept a METAR/DSM high when the 5-min feed is stale/gappy or sources agree
  FIX_F9   markets without strike metadata: parse strikes from the ticker, else UNTRADEABLE (never default YES)
  FIX_F10  new edge-after-fees trading rule (backtest / paper only); the old "YES top / NO all" rule is disabled
  FIX_F11  logic rule: the 6-hour max group / DSM rule out an impossible upper whole-F candidate of the known high

Overrides, in order of precedence:
  * in-process:  fixes.set_flags(FIX_F1=False, ...) or `with fixes.override(...)`  (used by the eval harness)
  * subprocess:  environment variable MODEL_FIXES='{"FIX_F1": false, ...}' (JSON), read at import time; the
                 production forecast runner passes the parent's flags to the model-script subprocesses this way.
"""
import contextlib
import json
import os

FLAG_NAMES = [f"FIX_F{i}" for i in range(1, 12)]

FIX_F1 = True
FIX_F2 = True
FIX_F3 = True
FIX_F4 = True
FIX_F5 = True
FIX_F6 = True
FIX_F7 = True
FIX_F8 = True
FIX_F9 = True
FIX_F10 = True
FIX_F11 = True

# F5: simulate every path to this Miami-local time (hour, minute). See eval/REPORT_v2.md for the late-high check.
F5_END_LOCAL = (19, 0)
# F10 defaults (cents). Live trading stays disabled regardless (kalshi_api/trading_guard.py).
F10_BUFFER_CENTS = 3
F10_HIGH_PRICE_CENTS = 90
F10_HIGH_PRICE_MIN_EDGE_CENTS = 5
F10_MAX_ENTRIES_PER_MARKET_DAY = 1
F10_VOLUME_LOOKBACK_MIN = 60


def _apply_env():
    raw = os.environ.get("MODEL_FIXES")
    if not raw:
        return
    try:
        data = json.loads(raw)
    except ValueError:
        return
    set_flags(**{k: bool(v) for k, v in data.items() if k in FLAG_NAMES})


def flags() -> dict:
    g = globals()
    return {k: bool(g[k]) for k in FLAG_NAMES}


def set_flags(**kw):
    g = globals()
    for k, v in kw.items():
        if k not in FLAG_NAMES:
            raise KeyError(f"unknown fix flag {k}")
        g[k] = bool(v)


def set_all(value: bool):
    set_flags(**{k: value for k in FLAG_NAMES})


@contextlib.contextmanager
def override(**kw):
    old = flags()
    set_flags(**kw)
    try:
        yield
    finally:
        set_flags(**old)


def env_for_subprocess() -> dict:
    """{"MODEL_FIXES": json} so a model-script subprocess sees the same flags as this process."""
    return {"MODEL_FIXES": json.dumps(flags())}


_apply_env()
