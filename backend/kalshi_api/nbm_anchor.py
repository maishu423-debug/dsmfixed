"""
F5: anchor the expected remaining warming to the NBM max-temperature guidance. compose() finds the shift delta of the
simulated future maxima so that E[max(K, M + delta)] = w * NBM_adj + (1 - w) * E[max(K, M)], where NBM_adj is the
latest NBM daytime max available at the issue time (cycle + 1.5 h) plus its walk-forward bias, net of the expected
settlement residual, and never below E[K]. w (per issue time) is learned walk-forward.
"""
import math

import numpy as np

from .daily_high import _num, _sigmoid, max_dist, mean, int_pmf  # noqa: F401


# ═══════════════════════════════════════════════════════════════════ F5: NBM anchor


def anchor_shift(K, mix, target_L):
    """delta such that E[max(K, round(M + delta))] = target_L (bisection; E is increasing in delta)."""
    lo, hi = -15.0, 15.0
    f_lo = mean(max_dist(K, int_pmf(mix, lo)))
    f_hi = mean(max_dist(K, int_pmf(mix, hi)))
    if target_L <= f_lo:
        return lo
    if target_L >= f_hi:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mean(max_dist(K, int_pmf(mix, mid))) < target_L:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return 0.5 * (lo + hi)


