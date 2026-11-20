"""
F4: settlement layer. The CLI high is computed from continuous data and catches peaks that the 5-minute feed and
the METARs miss (METAR misses the CLI peak on 68% of days). R = CLI - L, where L is the exact live max (5-min T-group
/ METAR T-group) of the climate day, is learned from history (walk-forward) per cell (issue-time group, peak-likely-
passed flag, convection flag), shrunk to the pooled distribution, and convolved onto the simulated L in
daily_high.compose(). See daily_high's module docstring for how F2 and F4 combine without double counting.
"""
import math

import numpy as np

from .daily_high import _num, _sigmoid, max_dist, mean, int_pmf  # noqa: F401


# ═══════════════════════════════════════════════════════════════════ F4: settlement residual

R_SUPPORT = list(range(-2, 5))


def f4_cell(ctx):
    """(issue group, peak-likely-passed flag, convection flag) for the residual table."""
    h = ctx.get("issue") or "14:00"
    hh, mm = map(int, h.split(":"))
    t = hh + mm / 60
    grp = "early" if t <= 13.5 else ("mid" if t <= 15.0 else "late")
    since = _num(ctx.get("hours_since_max")) or 0.0
    drop = _num(ctx.get("drop_from_max_c")) or 0.0
    peak = int(since >= 1.0 and drop >= 1.0)
    conv = int(bool(ctx.get("conv_so_far")))
    return f"{grp}|{peak}|{conv}"


def fit_residuals(cells, r, alpha=30.0):
    """{"pooled": pmf, "cells": {cell: pmf}, "n": {...}} with each cell shrunk to the pooled pmf (alpha pseudo-obs)."""
    r = np.clip(np.asarray(r, int), R_SUPPORT[0], R_SUPPORT[-1])
    idx = {v: i for i, v in enumerate(R_SUPPORT)}
    tot = np.full(len(R_SUPPORT), 0.25)
    for v in r:
        tot[idx[int(v)]] += 1
    pooled = tot / tot.sum()
    out = {"pooled": pooled.tolist(), "cells": {}, "n": {}}
    cells = list(cells)
    for c in set(cells):
        cnt = np.zeros(len(R_SUPPORT))
        for cc, v in zip(cells, r):
            if cc == c:
                cnt[idx[int(v)]] += 1
        out["cells"][c] = ((cnt + alpha * pooled) / (cnt.sum() + alpha)).tolist()
        out["n"][c] = int(cnt.sum())
    return out


def residual_pmf(params, ctx):
    if not params:
        return {0: 1.0}
    p = params["cells"].get(f4_cell(ctx), params["pooled"])
    return {v: float(q) for v, q in zip(R_SUPPORT, p) if q > 0}


def add_residual(L, R):
    out = {}
    for l_, pl in L.items():
        for r, pr in R.items():
            out[l_ + r] = out.get(l_ + r, 0.0) + pl * pr
    return out


