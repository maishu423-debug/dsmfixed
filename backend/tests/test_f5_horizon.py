"""F5 (Monte Carlo part): simulate to 19:00 Miami time; the 1-hour models leave the ensemble."""
import numpy as np
import pandas as pd

from config import fixes
from kalshi_api import algorithm as alg
from weather import path_tools


def test_horizon_hours():
    t1230_edt = pd.Timestamp("2025-07-01 16:30")          # naive UTC, 12:30 EDT
    t1600_est = pd.Timestamp("2025-01-15 21:00")          # 16:00 EST
    with fixes.override(FIX_F5=True):
        assert path_tools.horizon_hours(t1230_edt, 2) == 7    # 6.5 h to 19:00, rounded up
        assert path_tools.horizon_hours(t1600_est, 3) == 3
        assert path_tools.horizon_hours(pd.Timestamp("2025-07-01 23:30"), 2) == 1   # after 19:00: at least 1 h
    with fixes.override(FIX_F5=False):
        assert path_tools.horizon_hours(t1230_edt, 2) == 2


def test_chain_var_extra_noop_and_shapes():
    vs = {"a": np.zeros(5), "b": np.ones(5)}
    assert path_tools.chain_var_extra(vs, None, 0, ["a", "b"], None, 5) is vs
    params = {"a": np.zeros(2), "B": np.eye(2) * 0.5, "Sigma": np.eye(2) * 1e-6, "k": 2}
    out = path_tools.chain_var_extra(vs, params, 2, ["a", "b"], {"a": (-9, 9), "b": (-9, 9)}, 5)
    assert np.allclose(out["b"], 0.25, atol=1e-2)


def test_one_hour_models_dropped_from_the_mix():
    res = {m: {"high_temps": [80], "high_probs": [1.0]} for m in
           ["accuweather_2h", "var_2h", "accuweather_3h", "var_3h"]}
    res.update({m: {"high_temps": [99], "high_probs": [1.0]} for m in ["accuweather_1h", "var_1h"]})
    with fixes.override(FIX_F5=True):
        assert alg._get_path_high_pdf(res) == {80: 1.0}
    with fixes.override(FIX_F5=False):
        assert 99 in alg._get_path_high_pdf(res)
