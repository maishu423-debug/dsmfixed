"""
F6: recalibration of the bucket probabilities: temperature scaling q ∝ p**a over the live buckets, one a per issue
time, fitted walk-forward (each test month uses only earlier months). a < 1 widens (v1 was overconfident, slope
0.50). The hand-tuned SIGMA_SCALE in the 3-hour scripts is left in place; recalibration absorbs it.
"""
import math

import numpy as np

from .daily_high import _num, _sigmoid, max_dist, mean, int_pmf  # noqa: F401


# ═══════════════════════════════════════════════════════════════════ F6: recalibration


def recalibrate(p, a, eps=1e-3):
    """Temperature scaling over the LIVE buckets (p > 0): q ∝ p**a; dead buckets stay 0."""
    p = np.asarray(p, float)
    live = p > 0
    if not live.any():
        return p
    q = np.zeros_like(p)
    q[live] = np.exp(a * np.log(np.clip(p[live], eps, None)))
    return q / q.sum()


def fit_temperature(P, Y, eps=1e-3, grid=None):
    """a maximizing the mean log score of the realized bucket (scored with the same 0.001 floor)."""
    grid = np.arange(0.2, 2.01, 0.02) if grid is None else grid
    best, best_ll = 1.0, -np.inf
    for a in grid:
        ll = 0.0
        for p, y in zip(P, Y):
            q = recalibrate(p, a, eps)
            q = np.clip(q, eps, None)
            q = q / q.sum()
            ll += math.log(q[y])
        if ll > best_ll:
            best, best_ll = float(a), ll
    return best


