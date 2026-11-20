"""
F8: the DSM_MAX_EXCESS_F guard. The values below are the real KMIA observations of 2024-08-21 (5-min feed outage:
METAR 91 F at 15:53Z while the feed restarted at 19:00Z showing 26 C = 78.8 F) and 2024-11-27 (erroneous DSM: 84 F
"at 04:18 LST" while the METARs read 67 F; CLI 81 F).
"""
from datetime import datetime, timedelta, timezone

import pytest

from config import fixes
from kalshi_api import nws

Z = timezone.utc


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(nws, "_dsm_fail_until", 0.0)
    monkeypatch.setattr(nws, "_nws_fail_until", 0.0)
    nws._cache.clear()
    monkeypatch.setattr(nws, "get_dsm", lambda: {"latest": None})
    yield
    nws._cache.clear()


def _c(f):
    return (f - 32) * 5 / 9


# 2024-08-21: METAR exact F at :53 (T-group), real values
AUG21 = [(5, 82.94), (6, 82.04), (7, 80.96), (8, 80.96), (9, 80.06), (10, 80.96), (11, 82.04), (12, 84.92),
         (13, 87.98), (14, 89.96), (15, 91.04), (16, 80.96), (17, 77.00), (18, 75.92), (19, 78.08)]
AUG21_NOW = datetime(2024, 8, 21, 20, 0, tzinfo=Z)          # 16:00 EDT


def _aug21(monkeypatch):
    rows = [(datetime(2024, 8, 21, h, 53, tzinfo=Z), _c(f)) for h, f in AUG21]
    monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: {"readings": rows, "stale": False})
    # the 5-min feed only resumed at 19:00Z (outage all morning and midday)
    return [(datetime(2024, 8, 21, 19, 0, tzinfo=Z) + timedelta(minutes=5 * i), 78.8) for i in range(10)]


def test_2024_08_21_v1_threw_the_91f_high_away(monkeypatch):
    feed = _aug21(monkeypatch)
    with fixes.override(FIX_F8=False):
        assert nws.resolve_known_high_detail(78.8, AUG21_NOW, hf_obs=feed)[0] == 78.0


def test_2024_08_21_f8_keeps_the_91f_metar_high(monkeypatch):
    feed = _aug21(monkeypatch)
    with fixes.override(FIX_F8=True):
        assert nws.resolve_known_high_detail(78.8, AUG21_NOW, hf_obs=feed)[0] == 91.0
        assert nws.last_floor_decision()["source"] == "nws_hourly"
        # also without the feed's timestamps: the neighbouring METARs (90 F, 81 F) support it
        nws._cache.clear()
        assert nws.resolve_known_high(78.8, AUG21_NOW) == 91.0


NOV27_NOW = datetime(2024, 11, 27, 21, 0, tzinfo=Z)          # 16:00 EST
NOV27 = [(5, 69.08), (6, 68.0), (7, 66.92), (8, 66.92), (9, 66.92), (10, 66.92), (11, 66.92), (12, 71.06),
         (13, 75.02), (14, 78.08), (15, 78.98), (16, 80.96), (17, 80.06), (18, 80.06), (19, 80.06), (20, 78.98)]


def _nov27(monkeypatch):
    rows = [(datetime(2024, 11, 27, h, 53, tzinfo=Z), _c(f)) for h, f in NOV27]
    monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: {"readings": rows, "stale": False})
    monkeypatch.setattr(nws, "get_dsm", lambda: {"latest": {
        "kind": "partial", "date": "2024-11-27", "high_f": 84, "high_time_lst": "04:18",
        "issued_at": "2024-11-27T20:11:00+00:00", "report_time_lst": "15:00"}})
    start = datetime(2024, 11, 27, 5, 0, tzinfo=Z)
    return [(start + timedelta(minutes=5 * i), 66.2 if i < 90 else 80.6) for i in range(190)]


def test_2024_11_27_erroneous_dsm_is_rejected_and_logged(monkeypatch, capsys):
    feed = _nov27(monkeypatch)
    with fixes.override(FIX_F8=True):
        high, _ = nws.resolve_known_high_detail(80.6, NOV27_NOW, hf_obs=feed)
    assert high == 81.0                                         # the METAR max (80.96 F), not the bad DSM 84 F
    rej = [r for r in nws.rejection_log() if r["date"] == "2024-11-27"]
    assert rej and rej[-1]["source"] == "dsm" and rej[-1]["value_f"] == 84.0
    assert "above every neighbouring observation" in rej[-1]["reason"]
    assert "REJECTED dsm 84.0F" in capsys.readouterr().out
    assert "rejected" in nws.last_floor_decision()["reason"]


def test_2024_11_27_v1_accepted_the_bad_dsm(monkeypatch):
    feed = _nov27(monkeypatch)
    with fixes.override(FIX_F8=False):
        assert nws.resolve_known_high_detail(80.6, NOV27_NOW, hf_obs=feed)[0] == 84.0


def test_two_sources_agreeing_are_accepted_even_far_above_the_feed(monkeypatch):
    now = datetime(2026, 11, 5, 18, 0, tzinfo=Z)
    rows = [(now - timedelta(minutes=60), _c(91.94))]           # 92 F report
    monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: {"readings": rows, "stale": False})
    monkeypatch.setattr(nws, "get_dsm", lambda: {"latest": {"kind": "partial", "date": "2026-11-05", "high_f": 91,
                                                            "issued_at": "x", "report_time_lst": "12:00"}})
    feed = [(now - timedelta(minutes=5 * i), 80.6) for i in range(60)]
    with fixes.override(FIX_F8=True):
        assert nws.resolve_known_high_detail(80.6, now, hf_obs=feed)[0] == 92.0


def test_single_outlier_while_feed_runs_is_rejected(monkeypatch):
    now = datetime(2026, 11, 5, 18, 0, tzinfo=Z)
    t = now - timedelta(minutes=60)
    rows = [(t - timedelta(minutes=60), _c(80.06)), (t, _c(91.94)), (t + timedelta(minutes=60), _c(80.06))]
    monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: {"readings": rows, "stale": False})
    feed = [(now - timedelta(minutes=5 * i), 80.6) for i in range(60)]
    with fixes.override(FIX_F8=True):
        assert nws.resolve_known_high_detail(80.6, now, hf_obs=feed)[0] == 80.0


def test_stale_feed_accepts_the_metar_high(monkeypatch):
    now = datetime(2026, 11, 5, 18, 0, tzinfo=Z)
    rows = [(now - timedelta(minutes=30), _c(91.94))]
    monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: {"readings": rows, "stale": False})
    feed = [(now - timedelta(minutes=120 + 5 * i), 80.6) for i in range(20)]   # newest 5-min obs is 2 h old
    with fixes.override(FIX_F8=True):
        assert nws.resolve_known_high_detail(80.6, now, hf_obs=feed)[0] == 92.0


def test_six_hour_max_is_checked_against_its_whole_window(monkeypatch):
    """2023-05-25 pattern: 6-h max 88 F (= CLI) peaked early in the 12-18Z window; at 18Z it was much cooler."""
    from datetime import datetime as _dt
    now = _dt(2023, 5, 25, 19, 0, tzinfo=Z)
    ws, we = _dt(2023, 5, 25, 12, 0, tzinfo=Z), _dt(2023, 5, 25, 18, 0, tzinfo=Z)
    rows = [(_dt(2023, 5, 25, h, 53, tzinfo=Z), _c(f)) for h, f in ((13, 84.92), (14, 84.0), (16, 78.08), (17, 77.0))]
    data = {"readings": rows, "stale": False, "max6h": [(ws, we, 31.1, _dt(2023, 5, 25, 17, 53, tzinfo=Z))]}
    monkeypatch.setattr(nws, "_load_nws_readings", lambda now=None: data)
    feed = [(ws + timedelta(minutes=5 * i), 87.8 if i == 20 else 80.6) for i in range(84)]   # 31 C once, at 13:40Z
    with fixes.override(FIX_F3=True, FIX_F8=True):
        assert nws.resolve_known_high_detail(87.8, now, hf_obs=feed)[0] == 88.0
