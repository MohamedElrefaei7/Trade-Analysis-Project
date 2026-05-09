"""
seasonal_adjuster.py — subtract the recurring seasonal baseline.

Shipping has strong, predictable annual patterns — Chinese New Year
drains Chinese ports every Jan/Feb, the Sep/Oct peak season spikes
container imports, N-hemisphere summer softens bulk, Q4 inventory
builds lift air freight. Raw vessel counts or trade volumes can't be
compared year-over-year without stripping this out first.

STL (Seasonal-Trend decomposition using LOESS, via statsmodels) splits
a daily series into three additive pieces:

    raw = trend + seasonal + residual

`deseasonalize` returns trend + residual — the underlying level plus
genuine deviation from the expected seasonal baseline. A big October
vessel count is only flagged as a signal when it's above the October
baseline, not just because October is always busy.

STL needs at least ~2 full periods of history to be stable. With a
365-day period that's 730 days; we settle for a 90-day fallback which
captures the dominant sub-annual structure (weekly + within-quarter)
even if it can't isolate the full annual cycle. Below MIN_HISTORY_DAYS
the function returns the input unchanged and flags it as un-adjusted.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import STL

MIN_HISTORY_DAYS = 90


def deseasonalize(daily: pd.DataFrame, period: int = 7) -> tuple[pd.DataFrame, bool]:
    """
    Remove the seasonal component from a daily series.

    Parameters
    ----------
    daily  : single-column DataFrame (value) indexed by calendar day.
    period : seasonal period in days. 7 = weekly cycle (default, the
             most reliable with 90 days of history); 365 = annual — only
             pass this with ≥2 years of data.

    Returns (adjusted_frame, was_adjusted). If the history is too short
    or the series is constant, returns the input unchanged with
    was_adjusted=False.
    """
    if daily.empty or len(daily) < MIN_HISTORY_DAYS:
        return daily, False

    series = daily["value"].interpolate(limit_direction="both")
    if series.nunique() < 2:
        return daily, False

    try:
        result = STL(series, period=period, robust=True).fit()
    except Exception:
        # STL can fail on degenerate inputs; fall through un-adjusted
        # rather than poisoning the feature table.
        return daily, False

    adjusted = (result.trend + result.resid).to_frame("value")
    adjusted.index.name = daily.index.name
    return adjusted, True
