"""
builder.py — compute lead-lag signals across (feature, target, window) triples.

For every combination of (feature_name, target_name, window_days):

  1. Restrict to the trailing `window_days` of overlapping observations.
  2. For each lag in `LAG_RANGE`, align feature[t-lag] with target[t] and
     compute Pearson r. The lag with maximum |Pearson r| wins.
  3. Filter to relationships strong enough to be worth tracking
     (|pearson_r| ≥ PEARSON_THRESHOLD and at least MIN_SAMPLES observations).
  4. Keep top N strongest survivors per (as_of_date, window_days) and compute
     a Granger-causality p-value for each.
  5. Upsert one row per (as_of_date, feature_name, target_name, window_days).

Lag convention (matches dashboard/correlation.py):
    Positive lag_days  →  feature LEADS target by `lag_days` days.
    Negative lag_days  →  feature LAGS target by |lag_days| days.

Granger test (statsmodels): we ask whether the feature Granger-causes the
target up to `max(1, |lag_days|)` lags. The minimum p-value across tested
lags is stored, so a small `granger_p` means the feature has predictive
content for the target beyond what the target's own past explains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sqlalchemy import text

from clients.base import Session, logger

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_GRANGER = True
except Exception:  # pragma: no cover — statsmodels is in requirements
    _HAS_GRANGER = False


# ── Configuration ────────────────────────────────────────────────────────────

WINDOWS_DAYS: tuple[int, ...] = (180, 720)
LAG_RANGE: tuple[int, ...] = tuple(range(-30, 31))     # ±30 trading days
MIN_SAMPLES = 30                                       # post-alignment minimum
PEARSON_THRESHOLD = 0.15                               # write floor
TOP_N_PER_WINDOW = 100                                 # cap rows per (as_of, window)
GRANGER_MAX_LAG_CAP = 14                               # never test more than this


@dataclass(frozen=True)
class _BestLag:
    lag: int
    pearson: float
    spearman: float
    sample_size: int


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_features_wide(session) -> pd.DataFrame:
    rows = session.execute(
        text(
            """
            SELECT date, feature_name, value
            FROM features
            WHERE value IS NOT NULL
            """
        )
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "feature_name", "value"])
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="feature_name", values="value").sort_index()


def _load_targets_wide(session) -> pd.DataFrame:
    rows = session.execute(
        text(
            """
            SELECT date, target_name, value
            FROM targets
            WHERE value IS NOT NULL
            """
        )
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "target_name", "value"])
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="target_name", values="value").sort_index()


# ── Lag scan ─────────────────────────────────────────────────────────────────

def _windowed(s: pd.Series, end: pd.Timestamp, window_days: int) -> pd.Series:
    start = end - pd.Timedelta(days=window_days)
    return s.loc[(s.index >= start) & (s.index <= end)].dropna()


def _scan_best_lag(
    feature: pd.Series, target: pd.Series, end: pd.Timestamp, window_days: int
) -> _BestLag | None:
    """Return the lag (in days) maximising |pearson_r| for this pair, or None
    if no lag yields a usable sample."""
    f = _windowed(feature, end, window_days)
    t = _windowed(target, end, window_days)
    if len(f) < MIN_SAMPLES or len(t) < MIN_SAMPLES:
        return None

    best: _BestLag | None = None
    for lag in LAG_RANGE:
        # f.shift(lag) puts feature[t-lag] at index t  ⇒  positive lag = feature leads
        f_aligned = f.shift(lag)
        merged = pd.concat(
            [f_aligned.rename("x"), t.rename("y")], axis=1, join="inner"
        ).dropna()
        if len(merged) < MIN_SAMPLES:
            continue

        x = merged["x"].to_numpy()
        y = merged["y"].to_numpy()
        if x.std() == 0 or y.std() == 0:
            continue

        pr, _ = pearsonr(x, y)
        if np.isnan(pr):
            continue

        if best is None or abs(pr) > abs(best.pearson):
            sr, _ = spearmanr(x, y)
            best = _BestLag(
                lag=lag,
                pearson=float(pr),
                spearman=float(sr) if not np.isnan(sr) else 0.0,
                sample_size=int(len(merged)),
            )

    return best


def _granger_p_value(
    feature: pd.Series, target: pd.Series, lag: int, end: pd.Timestamp, window_days: int
) -> float | None:
    """Run Granger causality (feature → target) up to `max(1, |lag|)` lags
    bounded by GRANGER_MAX_LAG_CAP. Returns the minimum p-value across lags."""
    if not _HAS_GRANGER:
        return None

    f = _windowed(feature, end, window_days)
    t = _windowed(target, end, window_days)
    pair = pd.concat(
        [t.rename("effect"), f.rename("cause")], axis=1, join="inner"
    ).dropna()
    if len(pair) < MIN_SAMPLES:
        return None

    max_lag = min(GRANGER_MAX_LAG_CAP, max(1, abs(lag)))
    try:
        result = grangercausalitytests(pair.values, maxlag=max_lag, verbose=False)
    except Exception as exc:
        logger.debug("signals: granger failed (%s) — pair n=%d, max_lag=%d",
                     exc, len(pair), max_lag)
        return None

    p_values = []
    for lag_key, (stats, _models) in result.items():
        try:
            p_values.append(stats["ssr_ftest"][1])
        except (KeyError, IndexError, TypeError):
            continue
    if not p_values:
        return None
    p_min = min(p_values)
    if np.isnan(p_min):
        return None
    return float(p_min)


# ── Persistence ──────────────────────────────────────────────────────────────

_UPSERT_SQL = text(
    """
    INSERT INTO signals
        (as_of_date, feature_name, target_name, window_days,
         lag_days, pearson_r, spearman_r, granger_p, sample_size)
    VALUES
        (:as_of_date, :feature_name, :target_name, :window_days,
         :lag_days, :pearson_r, :spearman_r, :granger_p, :sample_size)
    ON CONFLICT (as_of_date, feature_name, target_name, window_days) DO UPDATE
    SET lag_days    = EXCLUDED.lag_days,
        pearson_r   = EXCLUDED.pearson_r,
        spearman_r  = EXCLUDED.spearman_r,
        granger_p   = EXCLUDED.granger_p,
        sample_size = EXCLUDED.sample_size
    """
)


def _write(rows: list[dict]) -> int:
    if not rows:
        return 0
    with Session() as s:
        s.execute(_UPSERT_SQL, rows)
        s.commit()
    return len(rows)


# ── Public API ───────────────────────────────────────────────────────────────

def build(as_of: pd.Timestamp | None = None) -> dict[str, int]:
    """
    Compute signals for every (feature, target, window) triple.

    Args:
        as_of: snapshot date — the right edge of every rolling window. Defaults
               to the latest date present in both features and targets.

    Returns:
        {window_days: rows_written} so callers can see per-window yield.
    """
    with Session() as session:
        f_wide = _load_features_wide(session)
        t_wide = _load_targets_wide(session)

    if f_wide.empty or t_wide.empty:
        logger.warning("signals: features or targets empty — nothing to do")
        return {}

    if as_of is None:
        as_of = pd.Timestamp(min(f_wide.index.max(), t_wide.index.max())).normalize()
    else:
        as_of = pd.Timestamp(as_of).normalize()

    logger.info(
        "signals: as_of=%s  features=%d  targets=%d  windows=%s",
        as_of.date(), f_wide.shape[1], t_wide.shape[1], WINDOWS_DAYS,
    )

    summary: dict[str, int] = {}
    all_rows: list[dict] = []

    for window_days in WINDOWS_DAYS:
        window_rows: list[dict] = []
        skipped_thin = 0
        skipped_weak = 0

        for feat_name in f_wide.columns:
            f = f_wide[feat_name]
            for tgt_name in t_wide.columns:
                t = t_wide[tgt_name]
                best = _scan_best_lag(f, t, as_of, window_days)
                if best is None:
                    skipped_thin += 1
                    continue
                if abs(best.pearson) < PEARSON_THRESHOLD:
                    skipped_weak += 1
                    continue
                window_rows.append({
                    "as_of_date":   as_of.date(),
                    "feature_name": feat_name,
                    "target_name":  tgt_name,
                    "window_days":  window_days,
                    "lag_days":     best.lag,
                    "pearson_r":    best.pearson,
                    "spearman_r":   best.spearman,
                    "granger_p":    None,
                    "sample_size":  best.sample_size,
                    "_feature":     f,
                    "_target":      t,
                })

        # Top-N strongest per window
        window_rows.sort(key=lambda r: abs(r["pearson_r"]), reverse=True)
        kept = window_rows[:TOP_N_PER_WINDOW]

        # Granger only on the survivors — it's the expensive part.
        for row in kept:
            row["granger_p"] = _granger_p_value(
                row.pop("_feature"), row.pop("_target"),
                row["lag_days"], as_of, window_days,
            )

        # Drop carrier-only series fields from any remaining rows (we trimmed
        # to top-N already, but be explicit).
        for row in kept:
            row.pop("_feature", None)
            row.pop("_target", None)

        all_rows.extend(kept)
        summary[f"window_{window_days}d"] = len(kept)
        logger.info(
            "signals: window=%dd  kept=%d  weak=%d  thin=%d",
            window_days, len(kept), skipped_weak, skipped_thin,
        )

    n = _write(all_rows)
    logger.info("signals: %d rows upserted across %d windows", n, len(WINDOWS_DAYS))
    summary["total_rows"] = n
    return summary


def run_all() -> dict[str, int]:
    """Entry point for the nightly Prefect flow."""
    return build()


if __name__ == "__main__":
    run_all()
