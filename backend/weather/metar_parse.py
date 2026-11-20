"""
F3: exact temperatures from METAR / SPECI text (pure functions, no I/O).

  * Body group  "TT/DD"   whole degrees C ("M" = minus), e.g. "22/M01". A whole-C value is only known to within one or
                          two whole-F values (see weather/asos_resolution.whole_f_candidates).
  * T-group     "TsnTTTsnTdTdTd" in the remarks, tenths of a degree C, sign digit 1 = negative, e.g. "T02171006"
                = 21.7 C / -0.6 C. ASOS measures whole F; the T-group is that value converted to C and rounded to
                tenths, so it identifies the whole-F value exactly: F = round_half_up(C * 1.8 + 32) (the tenths
                rounding moves F by at most 0.09, verified on 36,835 KMIA reports).
  * 6-hour max "1snTTT"   in the remarks of the synoptic reports (KMIA: the :53 reports before 00, 06, 12, 18 UTC):
                          the maximum temperature of the preceding 6 hours, tenths of a degree C. It is published only
                          with that report, so it may be used only from the report's availability time, and it covers
                          the window (synoptic hour - 6 h, synoptic hour].
"""
import math
import re
from datetime import datetime, timedelta, timezone

_TGROUP = re.compile(r"(?<!\S)T([01])(\d{3})(?:([01])(\d{3}))?(?!\S)")
_MAX6 = re.compile(r"(?<!\S)1([01])(\d{3})(?!\S)")
_MIN6 = re.compile(r"(?<!\S)2([01])(\d{3})(?!\S)")
_BODY = re.compile(r"(?<!\S)(M?\d{2})/(M?\d{2})?(?!\S)")


def _signed_tenths(sign, digits):
    v = int(digits) / 10.0
    return -v if sign == "1" else v


def _split_rmk(raw):
    raw = raw or ""
    i = raw.find(" RMK ")
    return (raw, "") if i < 0 else (raw[:i], raw[i + 5:])


def parse_tgroup(raw):
    """(temp_c, dew_c) in tenths of a degree C from the T-group, or (None, None) if the group is missing."""
    _, rmk = _split_rmk(raw)
    m = _TGROUP.search(rmk)
    if not m:
        return None, None
    t = _signed_tenths(m.group(1), m.group(2))
    d = _signed_tenths(m.group(3), m.group(4)) if m.group(3) else None
    return t, d


def parse_body_temp(raw):
    """(temp_c, dew_c) whole degrees C from the body "TT/DD" group, or (None, None)."""
    body, _ = _split_rmk(raw)
    m = _BODY.search(body)
    if not m:
        return None, None

    def conv(s):
        return None if not s else (-int(s[1:]) if s.startswith("M") else int(s))
    return conv(m.group(1)), conv(m.group(2))


def parse_6h_max(raw):
    """6-hour maximum temperature (tenths of a degree C) from the 1snTTT remark group, or None."""
    _, rmk = _split_rmk(raw)
    m = _MAX6.search(rmk)
    return _signed_tenths(m.group(1), m.group(2)) if m else None


def parse_6h_min(raw):
    _, rmk = _split_rmk(raw)
    m = _MIN6.search(rmk)
    return _signed_tenths(m.group(1), m.group(2)) if m else None


def whole_f_from_tenths_c(c):
    """Exact whole-F value of a tenths-of-C reading (the T-group / 6-hour group)."""
    return int(math.floor(c * 1.8 + 32.0 + 0.5))


def exact_reading(raw):
    """
    (temp_c, exact) for one report: the T-group when present (exact=True), else the body whole-C value (exact=False,
    so only its lower whole-F candidate is a proven value). (None, False) when neither group parses.
    """
    t, _ = parse_tgroup(raw)
    if t is not None:
        return t, True
    b, _ = parse_body_temp(raw)
    return (float(b), False) if b is not None else (None, False)


def six_hour_coverage(report_utc: datetime):
    """
    F11: the period actually covered by a 1snTTT group: the 6 hours ENDING AT THE REPORT TIME (e.g. 11:53-17:53Z for
    the 17:53Z report). Verified on the archive: with this window, an exact 5-min reading exceeds the group's whole-F
    value in 1 of 4,718 windows; with the synoptic-hour window (12-18Z) in 150 (eval/REPORT_v2_1.md).
    """
    t = report_utc.astimezone(timezone.utc)
    return t - timedelta(hours=6), t


def six_hour_window(valid_utc: datetime):
    """
    (window_start, window_end) UTC of a 6-hour max group carried by a report issued at `valid_utc`: the window ends
    at the synoptic hour the report belongs to (a report at 17:53Z belongs to 18Z). None if not a synoptic report.
    """
    t = valid_utc.astimezone(timezone.utc)
    end = (t + timedelta(minutes=15)).replace(minute=0, second=0, microsecond=0)
    if end.hour % 6 != 0 or (end - t) > timedelta(minutes=15):
        return None
    return end - timedelta(hours=6), end
