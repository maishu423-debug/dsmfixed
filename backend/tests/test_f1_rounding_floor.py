"""F1: whole-F candidates of a whole-C reading, and the path start/floor logic shared by the six model scripts."""
import math

import numpy as np
import pytest

from config import fixes
from weather import path_tools
from weather.asos_resolution import (lower_bound_whole_f, lower_bound_whole_f_c, possible_whole_f,
                                     whole_f_candidates)


@pytest.mark.parametrize("c,expected", [(25, [77]), (26, [78, 79]), (27, [80, 81]), (28, [82, 83]), (29, [84, 85]),
                                        (30, [86]), (31, [87, 88]), (32, [89, 90]), (33, [91, 92]), (34, [93, 94]),
                                        (35, [95]), (0, [32]), (-1, [30, 31]), (-5, [23])])
def test_whole_f_candidates(c, expected):
    assert whole_f_candidates(c) == expected
    assert lower_bound_whole_f_c(c) == expected[0]


def test_candidates_cover_every_whole_f_exactly_once():
    seen = {}
    for c in range(-20, 50):
        for f in whole_f_candidates(c):
            assert f not in seen
            seen[f] = c
    for f in range(-3, 121):
        assert seen[f] == int(math.floor((f - 32) * 5 / 9 + 0.5))


def _old_possible_whole_f(temp_f):            # v1 implementation, kept here to prove the refactor is identical
    def rhu(x):
        return int(math.floor(x + 0.5))
    if temp_f is None:
        return []
    temp_f = float(temp_f)
    c = (temp_f - 32.0) * 5.0 / 9.0
    if not abs(c - round(c)) <= 0.02:
        return [rhu(temp_f)]
    c = round(c)
    base = int(math.floor(temp_f))
    return [t for t in range(base - 3, base + 4) if rhu((t - 32.0) * 5.0 / 9.0) == c]


def test_possible_whole_f_unchanged():
    vals = [c * 1.8 + 32 for c in range(-10, 45)] + [x / 100 for x in range(5000, 11000, 7)] + [None]
    for v in vals:
        assert possible_whole_f(v) == _old_possible_whole_f(v), v


def test_lower_bound_of_31c_reading_is_87():
    assert lower_bound_whole_f(31 * 1.8 + 32) == 87          # 87.8 F reading


def test_start_and_floor_v1_when_flag_off():
    with fixes.override(FIX_F1=False):
        T = path_tools.start_array(87.8, 5)
        assert T.tolist() == [87.8] * 5
        assert path_tools.init_run_max(T).tolist() == T.tolist()
        assert path_tools.floor_value(87.8, 86.0) == 87.8
        s, pm, floor, hist = path_tools.apply_floor(np.array([86.0]), np.array([87.9]), 87.8, 86.0)
        assert floor == 87.8 and hist is None and pm.tolist() == [87.9] and s.tolist() == [87.8]


def test_start_and_floor_f1():
    with fixes.override(FIX_F1=True):
        np.random.seed(1)
        before = np.random.get_state()[1].copy()
        T = path_tools.start_array(87.8, 20000)
        assert (np.random.get_state()[1] == before).all(), "must not consume the model's random stream"
        assert T.min() >= 86.5 and T.max() < 88.5
        assert abs(T.mean() - 87.5) < 0.03
        assert np.isneginf(path_tools.init_run_max(T)).all()
        assert path_tools.floor_value(87.8, None) == 87
        assert path_tools.floor_value(87.8, 86.0) == 87
        assert path_tools.floor_value(87.8, 88.0) == 88.0
        # a reading with tenths is exact: no jitter
        assert path_tools.start_array(87.08, 3).tolist() == [87.08] * 3
        s, pm, floor, hist = path_tools.apply_floor(np.array([85.0, 86.9]), np.array([86.7, 88.6]), 87.8, None)
        assert floor == 87 and pm.tolist() == [87.0, 88.6]
        assert hist["step"] == 0.1 and abs(sum(hist["p"]) - 1) < 1e-6 and hist["lo"] == pytest.approx(86.7)


def test_path_max_can_settle_on_lower_candidate():
    """A 31 C reading with falling temperatures must leave probability on 87 F (v1 gave it zero)."""
    with fixes.override(FIX_F1=True):
        np.random.seed(3)
        T = path_tools.start_array(87.8, 10000)
        run = path_tools.init_run_max(T)
        for _ in range(4):
            T = T - 0.3 + 0.2 * np.random.randn(len(T))
            np.maximum(run, T, out=run)
        _, pm, _, _ = path_tools.apply_floor(T, run, 87.8, None)
        high = np.floor(pm + 0.5)
        assert (high == 87).mean() > 0.3
    with fixes.override(FIX_F1=False):
        np.random.seed(3)
        T = path_tools.start_array(87.8, 10000)
        run = path_tools.init_run_max(T)
        _, pm, _, _ = path_tools.apply_floor(T, run, 87.8, None)
        assert (np.floor(pm + 0.5) == 87).mean() == 0.0
