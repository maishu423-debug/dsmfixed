"""
Shared Monte-Carlo path handling for the six model scripts (one implementation instead of six copies).

v1 (FIX_F1 off) behaviour, reproduced exactly:
    every path starts at T0 (the current 5-min reading in F, e.g. 31 C = 87.8 F), its running maximum includes T0,
    and the paths are floored at max(obs_high, T0). Because 87.8 rounds to 88, the daily-high PDF could never put
    any probability on 87 F, although a 31 C reading is just as consistent with 87 F (132 forecasts in v1 gave 0 to
    the value that settled).

F1 (FIX_F1 on):
    * the start value of each path is drawn uniformly over the continuous range consistent with the whole-F
      candidates of the whole-C reading ([86.5, 88.5) for 31 C); a reading with tenths is used as is.
      The OU/VAR temperature recursion is linear in T, so this is the same as simulating from each candidate.
      The draw uses a private generator seeded from (but not advancing) the global numpy stream, so the model's
      own random numbers are unchanged (common random numbers across variants).
    * the running maximum covers the FUTURE steps only; what has already been observed (including the current
      reading) enters through the floor, which is max(obs_high, lower_bound_whole_f(T0)), never the rounded T0.
    * the unfloored future-max distribution is also returned (fut_max_hist) so the ensemble layer
      (kalshi_api/daily_high.py) can combine it with the known-high distribution (F2), the NBM anchor (F5) and the
      settlement residual (F4).
"""
import numpy as np

from config import fixes
from weather.asos_resolution import is_whole_celsius, lower_bound_whole_f, whole_f_candidates

HIST_STEP_F = 0.1


def _private_rng():
    """Generator seeded from the current global numpy state without consuming any of its numbers."""
    key = np.random.get_state()[1]
    return np.random.default_rng([int(key[0]), int(key[1]), int(key[2]), 1])


def start_range(T0):
    """(lo, hi) continuous range of the true temperature consistent with the reading, or None if exact."""
    T0 = float(T0)
    if not is_whole_celsius(T0):
        return None
    cand = whole_f_candidates(round((T0 - 32.0) * 5.0 / 9.0))
    return min(cand) - 0.5, max(cand) + 0.5


def start_array(T0, N):
    """Initial temperature of each of the N paths."""
    if not fixes.FIX_F1:
        return np.full(N, float(T0), dtype=np.float64)
    rng_ = start_range(T0)
    if rng_ is None:
        return np.full(N, float(T0), dtype=np.float64)
    return _private_rng().uniform(rng_[0], rng_[1], N)


def init_run_max(T):
    """Running maximum before the first step: v1 includes the start value; F1 counts future steps only."""
    if not fixes.FIX_F1:
        return T.copy()
    return np.full_like(T, -np.inf)


def floor_value(T0, obs_high):
    if not fixes.FIX_F1:
        return max(obs_high, T0) if obs_high is not None else T0
    lb = lower_bound_whole_f(float(T0))
    return max(obs_high, lb) if obs_high is not None else lb


def fut_max_hist(path_max, step=HIST_STEP_F):
    """Histogram of the unfloored future path maximum: {"lo": first bin centre, "step": step, "p": [...]}."""
    x = np.asarray(path_max, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    idx = np.floor(x / step + 0.5).astype(np.int64)
    lo = int(idx.min())
    cnt = np.bincount(idx - lo)
    p = cnt / cnt.sum()
    return {"lo": round(lo * step, 6), "step": step, "p": [round(float(v), 6) for v in p]}


def apply_floor(samples, path_max, T0, obs_high):
    """
    Floor the final temperatures and the path maxima at what has already been observed.
    Returns (samples, path_max_floored, floor, fut_hist); fut_hist is None when FIX_F1 is off.
    """
    hist = fut_max_hist(path_max) if fixes.FIX_F1 else None
    floor = floor_value(T0, obs_high)
    samples = np.maximum(samples, floor)
    path_max = np.maximum(path_max, floor)
    return samples, path_max, floor, hist


# ── F5: horizon to the end of the afternoon ──────────────────────────────────

def horizon_hours(now_utc_naive, default_hours):
    """
    Whole hours to simulate. v1: the script's fixed HORIZON_HOURS. F5: until F5_END_LOCAL (19:00 Miami time),
    rounded UP to a whole hour (the scripts build hourly covariate waypoints), and never less than 1.
    """
    if not fixes.FIX_F5:
        return default_hours
    import math
    from datetime import datetime as _dt, timezone as _tz
    from zoneinfo import ZoneInfo
    now = _dt.fromtimestamp(float(now_utc_naive.timestamp()) if getattr(now_utc_naive, "tzinfo", None)
                            else now_utc_naive.replace(tzinfo=_tz.utc).timestamp(), tz=_tz.utc)
    loc = now.astimezone(ZoneInfo("America/New_York"))
    end = loc.replace(hour=fixes.F5_END_LOCAL[0], minute=fixes.F5_END_LOCAL[1], second=0, microsecond=0)
    hours = (end - loc).total_seconds() / 3600.0
    return max(1, int(math.ceil(hours - 1e-9)))


def chain_var_extra(var_samples, var_params, n_extra, keys, clips, n_paths):
    """F5: chain n_extra more VAR(1) steps on the per-path covariate draws (longer horizon = wider covariates)."""
    if n_extra <= 0:
        return var_samples
    a, B, Sigma = var_params["a"], var_params["B"], var_params["Sigma"]
    k_v = var_params["k"]
    X = np.column_stack([var_samples[k] for k in keys])
    for _ in range(n_extra):
        mean = a + X @ B.T
        try:
            eps = np.random.multivariate_normal(np.zeros(k_v), Sigma, size=n_paths)
        except np.linalg.LinAlgError:
            eps = np.random.randn(n_paths, k_v) * np.sqrt(np.diag(Sigma))
        X = mean + eps
        for i, key in enumerate(keys):
            lo, hi = clips[key]
            X[:, i] = np.clip(X[:, i], lo, hi)
    out = dict(var_samples)
    for i, key in enumerate(keys):
        out[key] = X[:, i]
    return out
