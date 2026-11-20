"""F5: NBM TXN parsing and its availability rule (cycle + 1.5 h)."""
from datetime import date, datetime, timezone

from weather.nbm import latest_txn_from_csv

CSV = """runtime,ftime,model,tmp,txn,station
2026-11-01 07:00:00,2026-11-02 00:00:00,NBS,80,84,KMIA
2026-11-01 13:00:00,2026-11-02 00:00:00,NBS,81,85,KMIA
2026-11-01 13:00:00,2026-11-01 21:00:00,NBS,82,,KMIA
2026-11-01 19:00:00,2026-11-02 00:00:00,NBS,81,86,KMIA
"""


def test_latest_usable_cycle():
    d = date(2026, 11, 1)
    assert latest_txn_from_csv(CSV, d, datetime(2026, 11, 1, 14, 29, tzinfo=timezone.utc)) == \
        (84.0, datetime(2026, 11, 1, 7, tzinfo=timezone.utc))          # 13Z cycle not yet available
    assert latest_txn_from_csv(CSV, d, datetime(2026, 11, 1, 14, 30, tzinfo=timezone.utc))[0] == 85.0
    assert latest_txn_from_csv(CSV, d, datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc))[0] == 86.0
    assert latest_txn_from_csv(CSV, date(2026, 11, 2), datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc)) == (None, None)
