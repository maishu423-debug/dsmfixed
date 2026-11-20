"""F5: NBM anchor of the remaining warming (kalshi_api/nbm_anchor.py)."""
import math

import numpy as np
import pytest

from kalshi_api import daily_high as DH


def _hist(values, step=0.1):
    from weather.path_tools import fut_max_hist
    return fut_max_hist(np.asarray(values, float), step)


def test_nbm_anchor_moves_mean_towards_nbm():
    ctx = {"hf_c_max": 28.0, "issue": "12:30", "known_bound": 82, "nbm": 88.0}
    hists = {"m": _hist(np.random.default_rng(2).normal(84.0, 1.0, 20000))}
    p0, _ = DH.compose(ctx, hists, {"m": 1.0})
    p1, info = DH.compose(ctx, hists, {"m": 1.0}, anchor={"w": 0.5, "bias": 0.0})
    assert DH.mean(p1) == pytest.approx(0.5 * 88 + 0.5 * DH.mean(p0), abs=0.05)
    assert info["shift"] > 0
