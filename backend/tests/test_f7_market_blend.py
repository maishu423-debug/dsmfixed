"""F7: market blend p_blend (kalshi_api/market_blend.py)."""
import math

import numpy as np
import pytest

from kalshi_api import daily_high as DH
from kalshi_api import market_blend as MB


def _hist(values, step=0.1):
    from weather.path_tools import fut_max_hist
    return fut_max_hist(np.asarray(values, float), step)


def test_blend_identity_and_dead():
    pm = np.array([0.0, 0.6, 0.4])
    q = MB.blend(np.array([0.0, 0.3, 0.7]), pm, (0.0, 1.0, 0.0), live=[False, True, True])
    assert q[0] == 0 and q[1:] == pytest.approx([0.6, 0.4], abs=1e-6)
