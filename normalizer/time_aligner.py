"""
time_aligner.py — put any series on a daily calendar index.

Rules:

  * Upsampling (quarterly / monthly → daily):
        forward-fill last known value until the next observation lands.
        A Q1 GDP value is held from its release date through every day
        of the quarter until the Q2 release replaces it.

  * Downsampling (sub-daily → daily):
        `agg="mean"` for continuous series (AIS-driven counts, wait hours)
        `agg="last"` for price-like series (BDI close, FX rates).

  * Gap handling:
        market-data series (is_market=True) forward-fill over weekends
        and holidays. Shipping / count series leave gaps as NaN — zero
        vessels is a genuine zero, not a missing value, and the caller
        decides how to treat NaN vs 0.

The module is pure pandas — no DB writes. `feature_builder` is the only
caller; it hands in a DataFrame and gets back a daily-indexed one.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Agg = Literal["mean", "last", "sum", "max"]


def to_daily(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    value_col: str = "value",
    agg: Agg = "mean",
    is_market: bool = False,
) -> pd.DataFrame:
    """
    Resample a two-column (timestamp, value) frame to a calendar-day index.

    Parameters
    ----------
    df         : pd.DataFrame with at least `ts_col` and `value_col`.
    agg        : aggregation for sub-daily input ("mean", "last", "sum", "max").
    is_market  : if True, forward-fill across weekends/holidays.

    Returns a DataFrame indexed by daily `DatetimeIndex` with a single
    column `value`. Empty input returns an empty frame with the right
    index type so callers can concat unconditionally.
    """
    if df.empty:
        return pd.DataFrame(columns=["value"],
                            index=pd.DatetimeIndex([], name="date"))

    s = (
        pd.to_datetime(df[ts_col], utc=True)
          .dt.tz_convert("UTC")
          .dt.normalize()   # drop intraday component, keep tz
    )
    frame = pd.DataFrame({"date": s, "value": df[value_col].astype(float)})

    grouped = frame.groupby("date")["value"]
    match agg:
        case "mean":  daily = grouped.mean()
        case "last":  daily = grouped.last()
        case "sum":   daily = grouped.sum()
        case "max":   daily = grouped.max()

    # Reindex to a contiguous daily calendar so downstream steps see every day.
    full_idx = pd.date_range(daily.index.min(), daily.index.max(),
                             freq="D", tz="UTC", name="date")
    daily = daily.reindex(full_idx)

    if is_market:
        daily = daily.ffill()
    else:
        # Upsample path: if the source was lower-than-daily (e.g. quarterly
        # GDP), the raw observations create large NaN stretches between
        # points — forward-fill those. True count-series gaps (a day with
        # zero events) should have been represented as 0 upstream, not
        # missing, so ffill here is safe.
        daily = daily.ffill()

    return daily.to_frame("value")
