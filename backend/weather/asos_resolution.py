"""
What a KMIA temperature reading can actually tell us, in whole degrees F (pure functions, no I/O).

ASOS measures in WHOLE degrees F. Two kinds of report carry that measurement:

  * 5-minute observations (the NOAA HFMETAR feed, and the 5-minute rows in the NWS API / Synoptic) are converted to
    WHOLE degrees C. One degree C is 1.8 F, so two neighbouring F values often give the same C value:
    84F and 85F are both 29C (84.2F), 82F and 83F are both 28C (82.4F), 87F and 88F are both 31C (87.8F).
  * Hourly METAR / SPECI reports also carry tenths of a degree C (the T-group), which identifies the F value exactly:
    29.4C is 85F.

So a whole-degree-C value is only known to within a set of one or two whole-F values, while a value with tenths is exact.
The 5-minute value is the station's own 5-minute average, rounded to whole F and then to whole C (verified against
the 1-minute ASOS record).

Rounding chain, derived from the data (eval/scripts/v2_f1_rounding.py -> eval/results/v2_f1_rounding.json):
  whole F (measured)  ->  C = round_half_up((F - 32) * 5/9)  ->  whole-C group of the 5-minute report
  * 327,137 archived 5-min records carry both the whole-C group and the T-group (tenths C, which identifies F
    exactly): round-to-nearest matches 99.975% (81 isolated glitches), truncation only 55.5%;
    double rounding through tenths gives the same result as direct rounding (no x.45-x.5 cases exist).
  * 36,835 METARs (body whole C vs T-group): round-to-nearest matches 99.95%.
  * (F - 32) * 5/9 never ends in exactly .5, so there are no ties and the half-up/half-even choice does not matter.
  Hence whole_f_candidates(c) = {F : round((F - 32) * 5/9) == c}: one or two values (two for 79% of the C values
  between 10 and 38 C, e.g. 31 C -> {87, 88}; 30 C -> {86}).
"""
import math

WHOLE_C_TOLERANCE = 0.02       # a reading rounded to 2 decimals in F is within 0.005 C of a whole-degree-C value


def _round_half_up(x):
    return int(math.floor(x + 0.5))


def is_whole_celsius(temp_f):
    """True when the reading is (to within rounding) a whole number of degrees C, i.e. it has lost its precision."""
    c = (temp_f - 32.0) * 5.0 / 9.0
    return abs(c - round(c)) <= WHOLE_C_TOLERANCE


def whole_f_candidates(c_whole):
    """
    Every whole-degree-F value whose reading, converted to C and rounded to a whole degree, gives `c_whole`
    (sorted, one or two values). This is the rounding chain derived from the data (see the module docstring).
    """
    c = int(round(float(c_whole)))
    base = int(math.floor(c * 1.8 + 32.0))
    return [t for t in range(base - 3, base + 4) if _round_half_up((t - 32.0) * 5.0 / 9.0) == c]


def whole_f_candidates_tenths(c_tenths):
    """
    F11: every whole-F value consistent with a tenths-of-C value (T-group, 6-hour max group): the same rounding
    chain as whole_f_candidates, but to tenths: {F : round_half_up((F - 32) * 5/9, 1 decimal) == c_tenths}.
    Usually one value (29.4 C -> {85}); the set form keeps the rule exact if a value ever maps to two.
    """
    c10 = int(round(float(c_tenths) * 10))
    base = int(math.floor(float(c_tenths) * 1.8 + 32.0))
    return [t for t in range(base - 2, base + 3) if _round_half_up((t - 32.0) * 50.0 / 9.0) == c10]


def lower_bound_whole_f_c(c_whole):
    """min(whole_f_candidates(c_whole)): the lowest whole-F value consistent with a whole-C reading."""
    return min(whole_f_candidates(c_whole))


def possible_whole_f(temp_f):
    """
    The whole-degree-F values the station may have measured, given a reading in F.
    A whole-degree-C reading gives every F value that rounds to that C value (one or two); a reading with tenths
    of a degree C is exact and gives just its own rounded F. None gives [].
    """
    if temp_f is None:
        return []
    temp_f = float(temp_f)
    if not is_whole_celsius(temp_f):
        return [_round_half_up(temp_f)]
    return whole_f_candidates(round((temp_f - 32.0) * 5.0 / 9.0))


def lower_bound_whole_f(temp_f):
    """
    The lowest whole-degree-F value consistent with the reading, or None. `temp_f` is the reading as the feed delivers
    it, in F (a whole-C reading arrives as e.g. 87.8); for a whole-C reading this equals
    lower_bound_whole_f_c(c_whole) = min(whole_f_candidates(c_whole)).
    """
    possible = possible_whole_f(temp_f)
    return min(possible) if possible else None


def upper_bound_whole_f(temp_f):
    """The highest whole-degree-F value consistent with the reading, or None."""
    possible = possible_whole_f(temp_f)
    return max(possible) if possible else None
