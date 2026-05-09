"""
lag_adjuster.py — eliminate lookahead bias by shifting series forward
by their publication delay.

Every row in `economic_benchmarks` carries a `lag_days` value that
records how long after the observation date the figure actually became
available to a real trader. Q1 US GDP is dated March 31st but the BEA
advance release lands late April — so a backtest that reads the March
31st row pretending it was available on April 1st would cheat.

Given a daily-indexed series (see time_aligner.to_daily) and its
lag_days, `apply_lag` shifts the index forward by that many days so
each value sits on the date it was actually knowable.

Returns (shifted_series, was_adjusted: bool) so feature_builder can
stamp `lag_adjusted` on the resulting rows.
"""

from __future__ import annotations

import pandas as pd


def apply_lag(daily: pd.DataFrame, lag_days: int) -> tuple[pd.DataFrame, bool]:
    """
    Shift a daily frame forward by `lag_days`. Returns the shifted frame
    and a boolean flag indicating whether the shift was non-trivial
    (lag_days > 0 and the frame wasn't empty).
    """
    if daily.empty or lag_days is None or lag_days <= 0:
        return daily, False

    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=int(lag_days))
    shifted.index.name = daily.index.name
    return shifted, True
