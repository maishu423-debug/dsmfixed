"""
v2 daily-high composition: from the model scripts' simulated future maxima to a settlement (CLI) distribution and
bucket probabilities. Pure functions (numpy only), used by the live algorithm and by the evaluation harness.

    K   known exact live max so far (5-min feed exact F + METAR T-group max), a DISTRIBUTION (F2)
    M   max of the simulated paths over the remaining horizon (fut_max_hist from the scripts, F1/F5), optionally
        shifted by delta so its expectation is anchored to NBM (F5)
    L   = max(K, M)           end-of-day exact live max (what the 5-min feed / METARs would show)
    R   = CLI - L             settlement residual: the CLI is computed from continuous data and catches peaks the
                              5-min/METAR readings miss (F4); learned from history, conditioned on issue time, a
                              "peak likely passed" flag and a convection flag
    CLI = L + R, then folded at B, the proven lower bound on the CLI (resolve_known_high: METAR/DSM/6-hour max)

How F2 and F4 combine without double counting: F2 resolves only the whole-C -> whole-F ambiguity of the 5-min
feed's OWN maximum (its label is the exact F of the 5-min readings, from the archived T-group); F4's residual is
measured against the exact live max L (5-min exact + METAR exact), never against a rounded or lower-bound value.
So F2 moves mass between the two candidates of K, and F4 adds what no live reading saw.

F6 recalibration (temperature scaling per issue time) and F7 market blend act on the bucket probabilities.
"""
import math

import numpy as np

# ═══════════════════════════════════════════════════════════════════ helpers


def whole_f_candidates_c(c):
    c = int(round(float(c)))
    base = int(math.floor(c * 1.8 + 32.0))
    return [t for t in range(base - 3, base + 4) if int(math.floor((t - 32.0) * 5.0 / 9.0 + 0.5)) == c]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(X, y, l2=1.0, w=None, iters=50):
    """L2-penalized logistic regression by IRLS (intercept = column 0, not penalized). Returns coef array."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    beta = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        p = _sigmoid(X @ beta)
        W = w * p * (1 - p) + 1e-9
        g = X.T @ (w * (y - p)) - pen * beta
        H = (X * W[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


# ═══════════════════════════════════════════════════════════════════ F2: known high as a distribution

F2_FEATURES = ["const", "d_lo", "log_n_at_max", "met_minus_lo", "met_eq_lo", "hours_since_max", "dsm_le_lo"]
F2_DEFAULT = {"coef": [0.0] * len(F2_FEATURES), "features": F2_FEATURES, "n": 0}


def f2_candidates(ctx):
    """(lo, hi) whole-F candidates of the 5-min feed's whole-C maximum so far, or None."""
    c = ctx.get("hf_c_max")
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return None
    cand = whole_f_candidates_c(c)
    return min(cand), max(cand)


def f2_features(ctx, met_key="met_exact_max"):
    """Feature vector for P(the 5-min max is the UPPER candidate), or None when there is no ambiguity left."""
    lh = f2_candidates(ctx)
    if lh is None or lh[0] == lh[1]:
        return None
    lo, hi = lh
    met = _num(ctx.get(met_key))
    if met is not None and met >= hi:
        return None
    dsm = _num(ctx.get("dsm_high"))
    n = _num(ctx.get("n_at_max")) or 1.0
    since = _num(ctx.get("hours_since_max"))
    return np.array([1.0,
                     (lo - 32.0) * 5.0 / 9.0 - float(ctx["hf_c_max"]),
                     math.log1p(n),
                     max(-3.0, min(0.0, met - lo)) if met is not None else -3.0,
                     1.0 if met is not None and met == lo else 0.0,
                     min(6.0, since) if since is not None else 0.0,
                     1.0 if dsm is not None and dsm <= lo else 0.0])


def f2_predict(params, x):
    return float(_sigmoid(np.dot(np.asarray(params["coef"], float), x)))


def _ts(x):
    from datetime import datetime, timezone
    if x is None:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


def f11_upper_ruled_out(ctx):
    """
    F11 (logic rule, no fitted parameter): the UPPER whole-F candidate of the 5-min feed's maximum is impossible when
    EVERY 5-min reading at that maximum lies in a period whose proven maximum is below the upper value:
      * a published 6-hour max group (1snTTT, tenths C -> whole-F candidates) whose coverage, the 6 h ending at its
        report time, contains the reading's whole 5-minute averaging period;
      * today's DSM, which covers the climate day up to its report time.
    ctx: hf_c_max, max_times [ISO], six_groups [[report_utc ISO, tenths C], ...] (only groups already AVAILABLE at the
    issue time), dsm_high, dsm_report_utc.
    """
    from datetime import timedelta
    from weather.asos_resolution import whole_f_candidates_tenths
    lh = f2_candidates(ctx)
    times = [_ts(t) for t in (ctx.get("max_times") or [])]
    if lh is None or lh[0] == lh[1] or not times:
        return False
    hi = lh[1]
    covers = []
    for rep, c in ctx.get("six_groups") or []:
        rep = _ts(rep)
        top = max(whole_f_candidates_tenths(c))
        if top < hi:
            covers.append((rep - timedelta(hours=6), rep))
    dsm, drep = _num(ctx.get("dsm_high")), _ts(ctx.get("dsm_report_utc"))
    if dsm is not None and drep is not None and dsm < hi:
        covers.append((None, drep))
    if not covers:
        return False
    return all(any((lo is None or t - timedelta(minutes=5) >= lo) and t <= hi_t for lo, hi_t in covers)
               for t in times)


def known_high_dist(ctx, f2_params=None, met_key="met_exact_max", f11=False):
    """
    {whole F: probability} of K, the exact live max so far (5-min feed + METAR T-group). Without f2_params (F2 off)
    K is the proven lower bound: max(lower candidate, METAR exact max). With f11, a candidate that the 6-hour max
    group or the DSM makes impossible gets probability 0 (f11_upper_ruled_out).
    """
    lh = f2_candidates(ctx)
    met = _num(ctx.get(met_key))
    if lh is None:
        return {int(met): 1.0} if met is not None else {}
    lo, hi = lh
    base = lo if met is None else max(lo, int(met))
    if f2_params is None or lo == hi or base >= hi:
        return {int(base): 1.0}
    if f11 and f11_upper_ruled_out(ctx):
        return {int(base): 1.0}
    x = f2_features(ctx, met_key)
    p = f2_predict(f2_params, x) if x is not None else 0.0
    return {int(base): 1.0 - p, int(hi): p}


# ═══════════════════════════════════════════════════════════════════ future max (model scripts)


def mix_hists(hists, weights):
    """Weighted mixture of fut_max_hist dicts ({"lo","step","p"}) -> (lo, step, p) on a common grid."""
    items = [(hists[k], w) for k, w in weights.items() if k in hists and w > 0 and hists[k]]
    if not items:
        return None
    step = items[0][0]["step"]
    lo = min(round(h["lo"] / step) for h, _ in items)
    hi = max(round(h["lo"] / step) + len(h["p"]) for h, _ in items)
    p = np.zeros(hi - lo)
    tw = 0.0
    for h, w in items:
        i0 = round(h["lo"] / step) - lo
        p[i0:i0 + len(h["p"])] += w * np.asarray(h["p"], float)
        tw += w
    p /= p.sum()
    return lo * step, step, p


def int_pmf(mix, shift=0.0):
    """Integer pmf of round_half_up(M + shift), spreading each 0.1-F bin uniformly over its width."""
    lo, step, p = mix
    c = lo + step * np.arange(len(p)) + shift
    a, b = c - step / 2, c + step / 2
    n1 = np.floor(a + 0.5).astype(int)
    n2 = np.floor(b + 0.5 - 1e-12).astype(int)
    f1 = np.where(n1 == n2, 1.0, np.clip((n1 + 0.5 - a) / step, 0, 1))
    out = {}
    for k, m in zip(n1, p * f1):
        out[int(k)] = out.get(int(k), 0.0) + float(m)
    for k, m in zip(n2, p * (1 - f1)):
        if m > 0:
            out[int(k)] = out.get(int(k), 0.0) + float(m)
    return out


def max_dist(K, M):
    """pmf of max(K, M) for independent integer pmfs K, M."""
    ks = sorted(K)
    out = {}
    mk = np.array(sorted(M))
    mp = np.array([M[k] for k in mk])
    cdf = np.cumsum(mp)
    for k in ks:
        pk = K[k]
        below = cdf[mk <= k][-1] if (mk <= k).any() else 0.0
        out[k] = out.get(k, 0.0) + pk * below
        for m, q in zip(mk[mk > k], mp[mk > k]):
            out[int(m)] = out.get(int(m), 0.0) + pk * q
    return out


def mean(pmf):
    s = sum(pmf.values())
    return sum(k * v for k, v in pmf.items()) / s if s > 0 else float("nan")


def fold_at(pmf, bound):
    """The settlement cannot be below a proven bound: move the mass below it onto it."""
    if bound is None or (isinstance(bound, float) and math.isnan(bound)):
        return dict(pmf)
    b = int(math.floor(float(bound) + 0.5))
    out = {}
    for k, v in pmf.items():
        kk = max(int(k), b)
        out[kk] = out.get(kk, 0.0) + v
    return out


# ═══════════════════════════════════════════════════════════════════ full composition


def compose(ctx, hists, weights, *, f2_params=None, f4_params=None, anchor=None, met_key="met_exact_max",
            bound_key="known_bound", f11=False):
    """
    CLI pmf {whole F: p} for one forecast.
      ctx       dict: hf_c_max, n_at_max, hours_since_max, drop_from_max_c, met_exact_max, dsm_high, conv_so_far,
                issue, known_bound (proven lower bound B), nbm (point), ...
      hists     {model: fut_max_hist}; weights {model: weight}
      f2_params F2 logistic params, or None (K = proven lower bound)
      f4_params F4 residual table, or None (no settlement layer)
      anchor    {"w": weight on NBM, "bias": NBM bias (CLI - NBM)} or None (F5 anchor off)
    Returns (pmf, info).
    """
    mix = mix_hists(hists, weights)
    if mix is None:
        return {}, {"error": "no future-max histograms"}
    K = known_high_dist(ctx, f2_params, met_key, f11=f11)
    if not K:
        b = _num(ctx.get(bound_key))
        if b is None:
            return {}, {"error": "no known high"}
        K = {int(b): 1.0}
    R = None
    if f4_params:
        from .settlement import residual_pmf                     # F4
        R = residual_pmf(f4_params, ctx)
    shift = 0.0
    info = {}
    nbm = _num(ctx.get("nbm"))
    if anchor and nbm is not None and anchor.get("w", 0) > 0:
        L0 = max_dist(K, int_pmf(mix))
        eR = mean(R) if R else 0.0
        n_adj = max(nbm + anchor.get("bias", 0.0) - eR, mean(K))
        target = anchor["w"] * n_adj + (1 - anchor["w"]) * mean(L0)
        from .nbm_anchor import anchor_shift                      # F5
        shift = anchor_shift(K, mix, target)
        info.update(nbm_target_L=target, shift=shift)
    L = max_dist(K, int_pmf(mix, shift))
    if R:
        from .settlement import add_residual
        C = add_residual(L, R)
    else:
        C = L
    C = fold_at(C, _num(ctx.get(bound_key)))
    s = sum(C.values())
    C = {k: v / s for k, v in sorted(C.items()) if v > 1e-9}
    info.update(p_upper=K.get(max(K)) if len(K) > 1 else None, mean=mean(C))
    return C, info


def _num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v
