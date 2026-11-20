"""F2: the known high as a distribution over its whole-F candidates (kalshi_api/daily_high.py)."""
import math

import numpy as np
import pytest

from kalshi_api import daily_high as DH


def _hist(values, step=0.1):
    from weather.path_tools import fut_max_hist
    return fut_max_hist(np.asarray(values, float), step)


# ── F2 ──────────────────────────────────────────────────────────────────────

def test_known_high_without_f2_is_the_lower_bound():
    ctx = {"hf_c_max": 31.0, "n_at_max": 3, "met_exact_max": 86}
    assert DH.known_high_dist(ctx, None) == {87: 1.0}


def test_known_high_f2_splits_mass_over_candidates():
    ctx = {"hf_c_max": 31.0, "n_at_max": 3, "met_exact_max": 86, "hours_since_max": 2.0}
    p = DH.known_high_dist(ctx, DH.F2_DEFAULT)                    # default coef 0 -> P(upper) = 0.5
    assert p == {87: 0.5, 88: 0.5}


def test_known_high_exact_metar_resolves_ambiguity():
    assert DH.known_high_dist({"hf_c_max": 31.0, "met_exact_max": 88}, DH.F2_DEFAULT) == {88: 1.0}
    assert DH.known_high_dist({"hf_c_max": 30.0, "met_exact_max": 85}, DH.F2_DEFAULT) == {86: 1.0}   # 30 C = 86 only


def test_f2_logistic_fit_recovers_rate():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(4000), rng.normal(size=4000)])
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.3 + 1.2 * X[:, 1])))).astype(float)
    b = DH.fit_logistic(X, y, l2=0.0)
    assert abs(b[0] - 0.3) < 0.15 and abs(b[1] - 1.2) < 0.15


# ── composition ─────────────────────────────────────────────────────────────

def test_post_peak_upper_value_gets_its_probability():
    """After the peak (all future paths below the max) the upper candidate must get ~P(upper), not 0 (F2 test)."""
    ctx = {"hf_c_max": 34.0, "n_at_max": 4, "met_exact_max": 93, "hours_since_max": 3, "known_bound": 93}
    hists = {"m": _hist(np.full(1000, 85.0))}
    params = {"coef": [math.log(0.55 / 0.45)] + [0.0] * (len(DH.F2_FEATURES) - 1)}
    pmf, info = DH.compose(ctx, hists, {"m": 1.0}, f2_params=params)
    assert pmf[94] == pytest.approx(0.55, abs=1e-6) and pmf[93] == pytest.approx(0.45, abs=1e-6)
    pmf0, _ = DH.compose(ctx, hists, {"m": 1.0}, f2_params=None)
    assert pmf0 == {93: 1.0}                                       # v1 behaviour: upper value impossible


def test_int_pmf_matches_per_path_rounding():
    vals = np.random.default_rng(1).normal(88.0, 1.0, 50000)
    mix = DH.mix_hists({"m": _hist(vals)}, {"m": 1.0})
    pm = DH.int_pmf(mix)
    direct = {int(k): v / len(vals) for k, v in zip(*np.unique(np.floor(vals + 0.5), return_counts=True))}
    for k in range(85, 92):
        assert pm.get(k, 0) == pytest.approx(direct.get(k, 0), abs=0.01)


def test_max_dist_and_fold():
    L = DH.max_dist({87: 0.5, 88: 0.5}, {86: 0.2, 88: 0.3, 90: 0.5})
    assert L == pytest.approx({87: 0.1, 88: 0.4, 90: 0.5})
    assert DH.fold_at({85: 0.2, 87: 0.8}, 86) == {86: 0.2, 87: 0.8}
