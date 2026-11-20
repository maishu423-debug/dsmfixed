"""F6: walk-forward temperature scaling (kalshi_api/recalibration.py)."""
import math

import numpy as np
import pytest

from kalshi_api import daily_high as DH
from kalshi_api import recalibration as RC


def _hist(values, step=0.1):
    from weather.path_tools import fut_max_hist
    return fut_max_hist(np.asarray(values, float), step)


# ── F6 / F7 ─────────────────────────────────────────────────────────────────

def test_recalibrate_keeps_dead_buckets_and_flattens():
    q = RC.recalibrate([0.0, 0.7, 0.2, 0.1], 0.5)
    assert q[0] == 0 and q.sum() == pytest.approx(1) and q[1] < 0.7


def test_fit_temperature_detects_overconfidence():
    rng = np.random.default_rng(3)
    P, Y = [], []
    for _ in range(600):
        p = np.array([0.9, 0.05, 0.05])
        P.append(p)
        Y.append(0 if rng.random() < 0.6 else rng.integers(1, 3))
    a = RC.fit_temperature(P, Y)
    assert a < 0.8
