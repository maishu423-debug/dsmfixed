"""
F7: second output p_blend: logit(p_blend) = a + b*logit(p_mid) + c*logit(p_model) per live bucket, dead buckets 0,
renormalized. (a, b, c) fitted walk-forward on earlier months (live buckets); missing bid = 0, missing ask = 1,
probabilities floored at 0.001 before the logit, exactly as the evaluation scores the mid.
"""
import math

import numpy as np

from .daily_high import _num, _sigmoid, max_dist, mean, int_pmf  # noqa: F401


# ═══════════════════════════════════════════════════════════════════ F7: market blend


def logit(p, eps=1e-3):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def blend(p_model, p_mid, coef, live=None, eps=1e-3):
    """logit(p_blend) = a + b*logit(p_mid) + c*logit(p_model) per bucket; dead buckets 0; renormalized."""
    a, b, c = coef
    z = a + b * logit(p_mid, eps) + c * logit(p_model, eps)
    q = _sigmoid(z)
    if live is not None:
        q = np.where(np.asarray(live, bool), q, 0.0)
    s = q.sum()
    return q / s if s > 0 else q


