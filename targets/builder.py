"""
builder.py — compute prediction targets from the `features` table.

Each TargetSpec defines:
    name             — written into targets.target_name
    source_feature   — the feature_name to read from `features`
    horizon_days     — N-day forward log return horizon

Forward log return at date t is:    r_t = ln( v_{t+N} / v_t )

Source data is the daily-aligned `features` table (not raw economic_benchmarks)
so we automatically pick up the normalizer's calendar handling. If a source
feature has zero rows or fewer than `horizon_days + 1` observations, the spec
is logged and skipped — it will start producing rows on the next run once
upstream ingest catches up.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text

from clients.base import Session, logger


@dataclass(frozen=True)
class TargetSpec:
    name: str
    source_feature: str
    horizon_days: int


SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(name="BDI.fwd_return_5d",  source_feature="BDI.daily_close", horizon_days=5),
    TargetSpec(name="BDI.fwd_return_20d", source_feature="BDI.daily_close", horizon_days=20),
    TargetSpec(name="WCI.fwd_return_20d", source_feature="WCI.composite",   horizon_days=20),
)


_UPSERT_SQL = text(
    """
    INSERT INTO targets (date, target_name, value, horizon_days)
    VALUES (:date, :name, :value, :horizon)
    ON CONFLICT (date, target_name, horizon_days)
    DO UPDATE SET value = EXCLUDED.value
    """
)


def _load_feature(session, feature_name: str) -> pd.Series:
    """Return a date-indexed Series of `value` for one feature, sorted."""
    rows = session.execute(
        text(
            """
            SELECT date, value
            FROM features
            WHERE feature_name = :n AND value IS NOT NULL
            ORDER BY date
            """
        ),
        {"n": feature_name},
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(
        {pd.Timestamp(r.date): float(r.value) for r in rows}, name=feature_name
    )
    s.index = s.index.normalize()
    return s


def _forward_log_return(series: pd.Series, horizon: int) -> pd.Series:
    """log(v[t+N] / v[t]). Drops non-finite values (zeros, negatives, gaps)."""
    if series.empty or len(series) <= horizon:
        return pd.Series(dtype=float)
    fwd = series.shift(-horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(fwd / series)
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    return ret


def _build_one(session, spec: TargetSpec) -> int:
    src = _load_feature(session, spec.source_feature)
    if src.empty:
        logger.warning(
            "targets: source feature %r has no rows — skipping %s",
            spec.source_feature, spec.name,
        )
        return 0
    if len(src) <= spec.horizon_days:
        logger.warning(
            "targets: %s needs > %d obs of %r, have %d — skipping",
            spec.name, spec.horizon_days, spec.source_feature, len(src),
        )
        return 0

    ret = _forward_log_return(src, spec.horizon_days)
    if ret.empty:
        logger.info("targets: %s produced 0 rows", spec.name)
        return 0

    payload = [
        {
            "date": idx.date(),
            "name": spec.name,
            "value": float(val),
            "horizon": spec.horizon_days,
        }
        for idx, val in ret.items()
    ]
    session.execute(_UPSERT_SQL, payload)
    session.commit()
    logger.info(
        "targets: %s — %d rows  (%s..%s)",
        spec.name, len(payload), payload[0]["date"], payload[-1]["date"],
    )
    return len(payload)


def run_all() -> dict:
    """Compute every TargetSpec. Returns {target_name: rows_written}."""
    summary: dict[str, int] = {}
    with Session() as session:
        for spec in SPECS:
            summary[spec.name] = _build_one(session, spec)
    total = sum(summary.values())
    logger.info("targets: run_all complete — %d rows total", total)
    return summary


if __name__ == "__main__":
    run_all()
