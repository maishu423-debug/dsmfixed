"""F4: settlement residual layer (kalshi_api/settlement.py)."""
import math

import numpy as np
import pytest

from kalshi_api import daily_high as DH
from kalshi_api import settlement as ST


def _hist(values, step=0.1):
    from weather.path_tools import fut_max_hist
    return fut_max_hist(np.asarray(values, float), step)


def test_residual_layer_and_bound():
    ctx = {"hf_c_max": 30.0, "issue": "15:00", "known_bound": 86}
    hists = {"m": _hist(np.full(100, 84.0))}
    f4 = ST.fit_residuals(["late|1|0"] * 100, [0] * 80 + [1] * 20, alpha=0.0)
    pmf, _ = DH.compose(ctx, hists, {"m": 1.0}, f4_params=f4)
    assert sum(pmf.values()) == pytest.approx(1.0)
    assert pmf[87] > 0.15 and min(pmf) == 86
