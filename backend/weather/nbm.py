"""
F5: latest NBM daytime max-temperature guidance for KMIA (NBS bulletin "TXN" valid at 00Z of the next day = the
06-18 LST daytime max), read from the Iowa Mesonet MOS archive - the same source the evaluation used.

Availability rule (no lookahead, as in the evaluation): a cycle is usable only from its runtime + 1.5 h.
"""
import csv
import io
from datetime import datetime, timedelta, timezone

import requests

IEM_MOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
AVAILABLE_AFTER = timedelta(hours=1.5)
USER_AGENT = "kalshi-weather-model/2.0"


def _parse_dt(s):
    return datetime.strptime(s.strip()[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def latest_txn_from_csv(text, day, now):
    """(txn F, runtime) of the newest NBS cycle usable at `now` with a daytime max for local `day`; (None, None)."""
    target = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(days=1)
    best = None
    for row in csv.DictReader(io.StringIO(text)):
        try:
            rt, ft = _parse_dt(row["runtime"]), _parse_dt(row["ftime"])
            txn = row.get("txn")
            if ft != target or not txn or rt + AVAILABLE_AFTER > now:
                continue
            if best is None or rt > best[1]:
                best = (float(txn), rt)
        except (KeyError, ValueError):
            continue
    return best if best else (None, None)


def fetch_latest_txn(day, now=None, station="KMIA", timeout=20):
    now = now or datetime.now(timezone.utc)
    p = {"station": station, "model": "NBS", "format": "csv",
         "sts": (now - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%MZ"), "ets": now.strftime("%Y-%m-%dT%H:%MZ")}
    r = requests.get(IEM_MOS, params=p, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return latest_txn_from_csv(r.text, day, now)
