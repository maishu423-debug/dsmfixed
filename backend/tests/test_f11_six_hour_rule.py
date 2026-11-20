"""
F11: the 6-hour max group (1snTTT) and the DSM rule out an impossible UPPER whole-F candidate of the known high.
Every METAR below is a real KMIA report from the evaluation cache (eval/data/clean/metar.parquet).
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from config import fixes
from kalshi_api import daily_high as DH
from kalshi_api import nws
from weather import metar_parse as MP
from weather.asos_resolution import whole_f_candidates, whole_f_candidates_tenths

Z = timezone.utc

# (report time, raw METAR, whole-C 5-min max whose candidates are tested, expected: upper ruled out?)
REAL = [
    ("2024-07-11T11:53", "KMIA 111153Z 16003KT 10SM FEW040 BKN140 OVC250 29/20 A3000 RMK AO2 SLP159 T02890200 10289 20272 53014 $", 29, True),    # 6h max 84 F; {84,85}: 85 impossible
    ("2026-08-19T11:53", "KMIA 191153Z 31005KT 10SM FEW300 28/26 A3002 RMK AO2 SLP164 T02780256 10289 20272 52011 $", 29, True),               # 84 F
    ("2026-07-04T11:53", "KMIA 041153Z 00000KT 10SM FEW028 FEW050 BKN250 29/24 A3003 RMK AO2 SLP170 70006 T02890244 10289 20267 53006 $", 29, True),  # 84 F
    ("2025-12-21T11:53", "KMIA 211153Z 00000KT 10SM FEW025 21/19 A3014 RMK AO2 SLP205 T02110194 10239 20206 53009 $", 24, True),             # 75 F; {75,76}
    ("2023-05-07T11:53", "KMIA 071153Z 04005KT 10SM SCT026 BKN041 BKN055 25/21 A3010 RMK AO2 RAE09 SLP192 P0000 60000 70010 T02500211 10256 20244 53021", 26, True),  # 78 F; {78,79}
    ("2025-11-01T17:53", "KMIA 011753Z VRB05KT 10SM FEW040 SCT080 27/13 A3004 RMK AO2 SLP174 T02670133 10278 20183 58017 $", 28, True),     # 82 F; {82,83}
    ("2025-06-24T17:53", "KMIA 241753Z 09013G20KT 10SM SCT038 BKN050 BKN085 BKN250 32/22 A3015 RMK AO2 SLP210 60000 T03220217 10322 20278 58007 $", 32, False),  # 90 F; {89,90}: 90 possible
    ("2026-08-01T17:53", "KMIA 011753Z 14012G17KT 10SM FEW040 SCT070 BKN250 33/25 A2998 RMK AO2 SLP152 T03280250 10344 20283 58010", 34, False),  # 94 F; {93,94}
    ("2024-09-04T11:53", "KMIA 041153Z 00000KT 10SM SCT017 SCT200 28/27 A3001 RMK AO2 SLP161 60033 70057 T02830267 10283 20261 53008 $", 28, False),  # 83 F; {82,83}
    ("2025-07-15T11:53", "KMIA 151153Z 20003KT 10SM FEW023 SCT160 BKN200 26/23 A3003 RMK AO2 SLP169 CB DSNT NE MOV N 70025 T02610233 10261 20239 53006 $", 26, False),  # 79 F
    ("2026-09-08T11:53", "KMIA 081153Z 04003KT 10SM FEW025 FEW250 27/24 A3003 RMK AO2 SLP168 CB DSNT SE-S AND W 70015 T02670244 10272 20256 53017 $", 27, False),  # 81 F; {80,81}
    ("2025-06-02T17:53", "KMIA 021753Z 35010G15KT 7SM -TSRA SCT010 BKN035CB OVC060 24/22 A3000 RMK PK WND 17043/1737 RAB28 TSB13 OCNL LTGICCG W-N-E TS W-N-E MOV NE P0039 T02390228 10333 20239 55007", 33, False),  # 92 F; {91,92}
    ("2024-04-14T11:53", "KMIA 141153Z VRB03KT 10SM BKN045 21/14 A3018 RMK AO2 SLP221 T02060144 10211 20194 53015 $", 21, False),     # 70 F; {69,70}
    ("2023-05-31T17:53", "KMIA 311753Z 12010KT 10SM FEW027 FEW180 SCT210 BKN250 30/23 A2988 RMK AO2 SLP119 CB DSNT SW-W MOV N T03000233 10300 20267 58006 $", 29, False),  # 86 F >= 85
]


def _ctx(rep, raw, c, reading_at, groups=True):
    rep_t = datetime.fromisoformat(rep).replace(tzinfo=Z)
    return {"hf_c_max": float(c), "n_at_max": 1, "max_times": [reading_at(rep_t).isoformat()],
            "six_groups": [[rep_t.isoformat(), MP.parse_6h_max(raw)]] if groups else [], "met_exact_max": None}


@pytest.mark.parametrize("rep,raw,c,ruled", REAL)
def test_real_six_hour_groups(rep, raw, c, ruled):
    inside = _ctx(rep, raw, c, lambda t: t - timedelta(hours=2))
    assert DH.f11_upper_ruled_out(inside) is ruled
    p = DH.known_high_dist(inside, DH.F2_DEFAULT, f11=True)
    lo, hi = min(whole_f_candidates(c)), max(whole_f_candidates(c))
    if ruled:
        assert p == {lo: 1.0}
    else:
        assert p.get(hi, 0) > 0
    # F11 off: the v2 behaviour (P(upper) from F2 regardless of the 6-hour max)
    assert DH.known_high_dist(inside, DH.F2_DEFAULT, f11=False).get(hi, 0) > 0


def test_reading_outside_the_covered_window_is_not_ruled_out():
    rep, raw, c, _ = REAL[0]
    after = _ctx(rep, raw, c, lambda t: t + timedelta(minutes=10))               # after the report
    before = _ctx(rep, raw, c, lambda t: t - timedelta(hours=6, minutes=2))      # averaging period starts before
    assert not DH.f11_upper_ruled_out(after) and not DH.f11_upper_ruled_out(before)


def test_group_not_yet_published_is_not_used():
    rep, raw, c, _ = REAL[0]
    assert not DH.f11_upper_ruled_out(_ctx(rep, raw, c, lambda t: t - timedelta(hours=2), groups=False))


def test_all_readings_at_the_max_must_be_covered():
    rep, raw, c, _ = REAL[0]
    ctx = _ctx(rep, raw, c, lambda t: t - timedelta(hours=2))
    ctx["max_times"].append((datetime.fromisoformat(rep).replace(tzinfo=Z) + timedelta(minutes=20)).isoformat())
    assert not DH.f11_upper_ruled_out(ctx)


def test_dsm_rules_out_until_its_report_time():
    ctx = {"hf_c_max": 31.0, "max_times": ["2024-11-27T18:00:00+00:00"], "dsm_high": 87,
           "dsm_report_utc": "2024-11-27T20:00:00+00:00"}
    assert DH.f11_upper_ruled_out(ctx)                                           # 88 impossible, DSM says 87
    ctx["max_times"] = ["2024-11-27T20:30:00+00:00"]                              # reading after the DSM
    assert not DH.f11_upper_ruled_out(ctx)


def test_tenths_candidates_use_the_f1_chain():
    for c in np.arange(15.0, 38.0, 0.1):
        cand = whole_f_candidates_tenths(round(c, 1))
        assert all(round((f - 32) * 5 / 9 + 1e-9, 1) == round(c, 1) or abs((f - 32) * 5 / 9 - c) < 0.051 for f in cand)
    assert whole_f_candidates_tenths(28.9) == [84] and whole_f_candidates_tenths(30.0) == [86]


def test_six_hour_max_above_the_known_high_raises_it(monkeypatch):
    """2025-06-24 17:53Z: 6-hour max 32.2 C = 90 F while the 5-min feed max is 31 C (87/88): B rises to 90."""
    rep = datetime(2025, 6, 24, 17, 53, tzinfo=Z)
    raw = REAL[6][1]
    ws, we = MP.six_hour_window(rep)
    data = {"readings": [(rep, 32.2)], "max6h": [(ws, we, MP.parse_6h_max(raw), rep)], "stale": False}
    monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: data)
    monkeypatch.setattr(nws, "get_dsm", lambda: {"latest": None})
    monkeypatch.setattr(nws, "_nws_fail_until", 0.0)
    nws._cache.clear()
    with fixes.override(FIX_F3=True, FIX_F8=True, FIX_F11=True):
        assert nws.resolve_known_high(87.8, rep + timedelta(minutes=20)) == 90.0
    nws._cache.clear()
    # and the composed CLI distribution can never go below it
    from weather.path_tools import fut_max_hist
    ctx = {"hf_c_max": 31.0, "known_bound": 90, "max_times": [(rep - timedelta(hours=1)).isoformat()],
           "six_groups": [[rep.isoformat(), 32.2]]}
    pmf, _ = DH.compose(ctx, {"m": fut_max_hist(np.full(100, 86.0))}, {"m": 1.0}, f2_params=DH.F2_DEFAULT, f11=True)
    assert min(pmf) == 90
