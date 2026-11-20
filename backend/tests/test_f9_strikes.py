"""F9: markets without strike metadata are parsed from the ticker, else UNTRADEABLE - never YES by default."""
import pytest

from config import fixes
from kalshi_api.algorithm import _in_bracket
from kalshi_api.strikes import parse_ticker, resolve_strikes

SIBS = ["KXHIGHMIA-25JAN18-T74", "KXHIGHMIA-25JAN18-B74.5", "KXHIGHMIA-25JAN18-B76.5",
        "KXHIGHMIA-25JAN18-B78.5", "KXHIGHMIA-25JAN18-B80.5", "KXHIGHMIA-25JAN18-T81"]


def test_between_from_ticker():
    assert parse_ticker("KXHIGHMIA-25JAN16-B73.5") == (73.0, 74.0, "between")


def test_tail_side_from_rules_text():
    assert parse_ticker("KXHIGHMIA-25JAN18-T81", text="... is greater than 81, then ...") == (81.0, None, "greater")
    assert parse_ticker("KXHIGHMIA-25JAN23-T65", text="... is less than 65, then ...") == (None, 65.0, "less")


def test_tail_side_from_siblings():
    assert parse_ticker("KXHIGHMIA-25JAN18-T81", SIBS) == (81.0, None, "greater")
    assert parse_ticker("KXHIGHMIA-25JAN18-T74", SIBS) == (None, 74.0, "less")


def test_unparseable_is_untradeable():
    r = resolve_strikes({"ticker": "KXHIGHMIA-25JAN18-T81"})       # no text, no siblings
    assert r["tradeable"] is False and r["strike_type"] == "untradeable"
    r = resolve_strikes({"ticker": "WEIRD"})
    assert r["tradeable"] is False


def test_metadata_wins_when_present():
    r = resolve_strikes({"ticker": "X-B73.5", "floor_strike": 60, "cap_strike": 61, "strike_type": "between"})
    assert (r["floor"], r["cap"], r["strike_source"]) == (60.0, 61.0, "metadata")


def test_in_bracket_never_defaults_to_yes():
    with fixes.override(FIX_F9=True):
        assert _in_bracket(80, None, None) is False
    with fixes.override(FIX_F9=False):
        assert _in_bracket(80, None, None) is True                 # the v1 bug, kept for reproduction


@pytest.mark.parametrize("t,yes", [(81, False), (82, True), (95, True)])
def test_parsed_greater_tail_semantics(t, yes):
    f, c, _ = parse_ticker("KXHIGHMIA-25JAN18-T81", SIBS)
    assert _in_bracket(t, f, c) is yes
