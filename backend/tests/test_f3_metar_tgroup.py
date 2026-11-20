"""
F3: exact METAR T-group / 6-hour max parsing. All strings below are real KMIA reports taken verbatim from the
evaluation cache (eval/data/clean/metar.parquet, column `metar`).
"""
from datetime import datetime, timedelta, timezone

import pytest

from config import fixes
from kalshi_api import nws
from weather import metar_parse as MP

REAL = [
    # raw METAR,                                                                       T-group (t, td), body (t, td), 6h max
    ("KMIA 010053Z 12007KT 10SM CLR 24/20 A3018 RMK AO2 SLP220 T02440200 $", (24.4, 20.0), (24, 20), None),
    ("KMIA 211753Z 31011G17KT 10SM FEW030 22/M01 A3019 RMK AO2 SLP223 T02171006 10222 20133 58017",
     (21.7, -0.6), (22, -1), 22.2),                                                        # negative dew point
    ("KMIA 211853Z COR VRB06G14KT 10SM FEW030 22/M01 A3016 RMK AO2 SLP214 T02171011", (21.7, -1.1), (22, -1), None),
    ("KMIA 031553Z 36009G20KT 340V040 10SM FEW040 19/M02 A3035 RMK AO2 SLP276 T01941017", (19.4, -1.7), (19, -2), None),
    ("KMIA 010553Z 11004KT 10SM CLR 23/20 A3017 RMK AO2 SLP218 T02280200 10250 20228 58008 $",
     (22.8, 20.0), (23, 20), 25.0),
    ("KMIA 011153Z 00000KT 10SM SCT020 22/21 A3018 RMK AO2 SLP221 T02170211 10228 20206 53012 $",
     (21.7, 21.1), (22, 21), 22.8),
    ("KMIA 011753Z 11010KT 10SM SCT031 29/20 A3020 RMK AO2 SLP226 T02890200 10289 20217 58015 $",
     (28.9, 20.0), (29, 20), 28.9),
    ("KMIA 271753Z 09008KT 10SM FEW036 27/17 A3003 RMK AO2 SLP169 T02670167 10272 20200 58018 $",
     (26.7, 16.7), (27, 17), 27.2),
    # missing T-group (specials / some routine reports)
    ("KMIA 040947Z 34005KT 10SM SCT015 BKN110 18/15 A3016 RMK AO2 $", (None, None), (18, 15), None),
    ("KMIA 042347Z 05014G25KT 10SM BKN026 BKN037 BKN060 23/18 A3022 RMK AO2 $", (None, None), (23, 18), None),
    ("KMIA 271151Z 10SM FEW002 FEW030 19/18 A3005 RMK AO2 $", (None, None), (19, 18), None),
    ("KMIA 051451Z 22006KT 8SM SCT018 BKN035 BKN070 OVC200 22/21 A3017 RMK AO2 LTG DSNT NE CB DSNT NE-SE MOV NE",
     (None, None), (22, 21), None),
]


@pytest.mark.parametrize("raw,tg,body,mx", REAL)
def test_real_metars(raw, tg, body, mx):
    assert MP.parse_tgroup(raw) == tg
    assert MP.parse_body_temp(raw) == body
    assert MP.parse_6h_max(raw) == mx


def test_negative_temperature_groups():
    # KMIA never reports a negative temperature; the sign handling is checked on a real report with the sign flipped
    raw = "KMIA 211753Z 31011G17KT 10SM FEW030 M02/M05 A3019 RMK AO2 SLP223 T10221050 11011 21033 58017"
    assert MP.parse_tgroup(raw) == (-2.2, -5.0)
    assert MP.parse_body_temp(raw) == (-2, -5)
    assert MP.parse_6h_max(raw) == -1.1
    assert MP.parse_6h_min(raw) == -3.3
    assert MP.whole_f_from_tenths_c(-2.2) == 28


def test_missing_and_garbage():
    assert MP.parse_tgroup("") == (None, None)
    assert MP.parse_tgroup(None) == (None, None)
    assert MP.parse_body_temp("KMIA 042347Z AUTO RMK AO2") == (None, None)
    assert MP.exact_reading("KMIA 042347Z AUTO RMK AO2") == (None, False)
    # a 5-digit group starting with 1 in the BODY is not a 6-hour max
    assert MP.parse_6h_max("KMIA 271753Z 10011KT 10SM 27/17 A3003 RMK AO2") is None


def test_exact_reading_prefers_tgroup():
    assert MP.exact_reading(REAL[0][0]) == (24.4, True)
    assert MP.exact_reading(REAL[8][0]) == (18.0, False)


@pytest.mark.parametrize("c,f", [(29.4, 85), (29.7, 85), (30.6, 87), (31.1, 88), (32.2, 90), (33.9, 93), (21.7, 71)])
def test_whole_f_from_tenths(c, f):
    assert MP.whole_f_from_tenths_c(c) == f


def test_six_hour_window():
    t = datetime(2024, 11, 27, 17, 53, tzinfo=timezone.utc)
    ws, we = MP.six_hour_window(t)
    assert (ws.hour, we.hour) == (12, 18)
    assert MP.six_hour_window(datetime(2024, 11, 27, 16, 53, tzinfo=timezone.utc)) is None


# ── wired into resolve_known_high (F3 on) ─────────────────────────────────────

NOW = datetime(2026, 11, 5, 19, 30, tzinfo=timezone.utc)      # 2:30 PM EST


@pytest.fixture
def isolate(monkeypatch):
    monkeypatch.setattr(nws, "_dsm_fail_until", 0.0)
    monkeypatch.setattr(nws, "_nws_fail_until", 0.0)
    nws._cache.clear()
    monkeypatch.setattr(nws, "get_dsm", lambda: {"latest": None})
    yield
    nws._cache.clear()


def _feature(when, raw):
    return {"properties": {"timestamp": when.isoformat(), "rawMessage": raw, "temperature": {"value": None}}}


def test_load_readings_uses_raw_tgroup_and_6h_max(monkeypatch, isolate):
    t1 = datetime(2026, 11, 5, 16, 53, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 5, 17, 53, tzinfo=timezone.utc)
    payload = {"features": [_feature(t1, "KMIA 051653Z 05006KT 10SM FEW035 27/17 A3007 RMK AO2 SLP181 T02720172 $"),
                            _feature(t2, REAL[7][0])]}

    class R:
        def json(self):
            return payload
    monkeypatch.setattr(nws, "_get", lambda *a, **k: R())
    with fixes.override(FIX_F3=True):
        data = nws._load_nws_readings(NOW)
        assert data["readings"] == [(t1, 27.2), (t2, 26.7)]
        (ws, we, c, avail), = data["max6h"]
        assert (ws.hour, we.hour, c, avail) == (12, 18, 27.2, t2)
        monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: data)
        # 5-min feed high 26 C = 78.8 F (78 or 79); T-group max 27.2 C = 81 F; 6-hour max 27.2 C = 81 F
        assert nws.resolve_known_high(78.8, NOW) == 81.0


def test_six_hour_max_raises_the_bound_and_respects_availability(monkeypatch, isolate):
    t = datetime(2026, 11, 5, 17, 53, tzinfo=timezone.utc)
    ws, we = MP.six_hour_window(t)
    data = {"readings": [(t, 26.1)], "max6h": [(ws, we, 27.8, t)], "stale": False}   # 6-h max 82 F, hourly 79 F
    monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: data)
    with fixes.override(FIX_F3=True):
        assert nws.resolve_known_high(78.8, NOW) == 82.0
        assert nws.last_floor_decision()["source"] == "metar_6h_max"
        nws._cache.clear()
        assert nws.resolve_known_high(78.8, t - timedelta(minutes=1)) == 79.0     # not yet published
    with fixes.override(FIX_F3=False):
        nws._cache.clear()
        assert nws.resolve_known_high(78.8, NOW) == 79.0
