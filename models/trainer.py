"""
trainer.py — fit one ElasticNet per (target, horizon) on lag-adjusted z-scores.

Pipeline per spec:
  1. Load the target series (forward log return) and the wide z-score matrix
     of every feature in `features` (rolling 90-day z-score, already lag-
     adjusted upstream).
  2. Restrict features to the target's date range, forward-fill ≤ MAX_FFILL_DAYS,
     drop features whose coverage of the target's dates is below
     MIN_FEATURE_COVERAGE, then drop dates with any remaining NaN.
  3. Walk-forward OOS predictions on the realised period:
       TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=horizon_days)
     The `gap` removes overlap between training targets (which see up to
     train_end + horizon) and test targets (which start at train_end + 1 + gap).
     Each fold fits a fresh ElasticNetCV with its own inner 3-fold TimeSeriesSplit.
  4. Live predictions on dates past the last realised target: fit a final
     ElasticNetCV on all realised data, predict on every date in the live span
     where the same kept features are present.
  5. predicted_z is the z-score of `predicted_value` against the historical
     OOS prediction distribution — that's "how unusual is today's call vs the
     model's typical output".

`model_version` is bumped whenever feature engineering or hyperparameters
change. The UNIQUE constraint on (date, target_name, horizon_days, model_version)
means re-runs overwrite cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import text

from clients.base import Session, logger


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_VERSION = "elasticnet_v1"

MIN_FEATURE_COVERAGE = 0.5      # feature must cover ≥ 50% of target dates
FFILL_DAYS = 180                # mixed-cadence features (daily BDI, weekly WCI,
                                # monthly port, quarterly GDP) need a generous
                                # forward-fill so a slow feature doesn't shrink
                                # the training matrix or block live predictions.
                                # No leakage — ffill only carries past values.
MIN_TRAIN_SAMPLES = 60          # minimum (X, y) rows to fit anything

N_SPLITS_CV = 5                 # outer walk-forward folds
INNER_CV_SPLITS = 3             # inner CV (alpha/l1_ratio selection)
N_ALPHAS = 20
L1_RATIOS = (0.1, 0.5, 0.9)
MAX_ITER = 5_000
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    target_name: str
    horizon_days: int


SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(target_name="BDI.fwd_return_5d",  horizon_days=5),
    ModelSpec(target_name="BDI.fwd_return_20d", horizon_days=20),
    ModelSpec(target_name="WCI.fwd_return_20d", horizon_days=20),
)


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_target(session, name: str, horizon: int) -> pd.Series:
    rows = session.execute(
        text(
            """
            SELECT date, value
            FROM targets
            WHERE target_name = :n
              AND horizon_days = :h
              AND value IS NOT NULL
            ORDER BY date
            """
        ),
        {"n": name, "h": horizon},
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(
        {pd.Timestamp(r.date).normalize(): float(r.value) for r in rows},
        name=name,
    ).sort_index()
    return s


def _load_features_z(session) -> pd.DataFrame:
    rows = session.execute(
        text(
            """
            SELECT date, feature_name, z_score
            FROM features
            WHERE z_score IS NOT NULL
            """
        )
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "feature_name", "z_score"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return (
        df.pivot(index="date", columns="feature_name", values="z_score")
          .sort_index()
    )


# ── Matrix construction ──────────────────────────────────────────────────────

def _select_features(target: pd.Series, features: pd.DataFrame) -> list[str]:
    """Features that cover ≥ MIN_FEATURE_COVERAGE of the target's date range
    after a generous forward-fill."""
    if target.empty or features.empty:
        return []
    start, end = target.index.min(), target.index.max()
    F = features.loc[(features.index >= start) & (features.index <= end)]
    F = F.reindex(F.index.union(target.index)).sort_index().ffill(limit=FFILL_DAYS)
    F_at_t = F.reindex(target.index)
    coverage = F_at_t.notna().mean(axis=0)
    return [c for c, frac in coverage.items() if frac >= MIN_FEATURE_COVERAGE]


def _aligned_matrix(
    dates: pd.DatetimeIndex, features: pd.DataFrame, kept: list[str]
) -> pd.DataFrame:
    """Forward-fill kept features within FFILL_DAYS and align to `dates`.
    NaN rows dropped by caller."""
    if not kept:
        return pd.DataFrame()
    F = features[kept].copy()
    F = F.reindex(F.index.union(dates)).sort_index().ffill(limit=FFILL_DAYS)
    return F.reindex(dates)


def _new_estimator(horizon: int) -> ElasticNetCV:
    inner_cv = TimeSeriesSplit(n_splits=INNER_CV_SPLITS, gap=horizon)
    return ElasticNetCV(
        l1_ratio=list(L1_RATIOS),
        alphas=N_ALPHAS,
        cv=inner_cv,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


# ── Fit / predict ────────────────────────────────────────────────────────────

def _walk_forward_oos(
    X: pd.DataFrame, y: pd.Series, horizon: int
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Manual walk-forward — TimeSeriesSplit doesn't partition all samples,
    so cross_val_predict isn't directly applicable. We fit a fresh estimator
    on each train fold and predict on its test fold."""
    n_splits = min(N_SPLITS_CV, max(2, len(X) // max(horizon * 2, 1)))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

    oos = np.full(len(X), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < MIN_TRAIN_SAMPLES:
            continue
        est = _new_estimator(horizon)
        est.fit(X.values[train_idx], y.values[train_idx])
        oos[test_idx] = est.predict(X.values[test_idx])

    mask = ~np.isnan(oos)
    return X.index[mask], oos[mask]


def _persist_predictions(rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = text(
        """
        INSERT INTO predictions
            (date, target_name, horizon_days,
             predicted_value, predicted_z, model_version)
        VALUES
            (:date, :target_name, :horizon_days,
             :predicted_value, :predicted_z, :model_version)
        ON CONFLICT (date, target_name, horizon_days, model_version) DO UPDATE
        SET predicted_value = EXCLUDED.predicted_value,
            predicted_z     = EXCLUDED.predicted_z
        """
    )
    with Session() as s:
        s.execute(sql, rows)
        s.commit()
    return len(rows)


# ── Per-spec orchestration ───────────────────────────────────────────────────

def _build_one(spec: ModelSpec, features_z: pd.DataFrame) -> int:
    with Session() as session:
        y_full = _load_target(session, spec.target_name, spec.horizon_days)

    if y_full.empty:
        logger.warning("models: %s — target has no rows, skipping", spec.target_name)
        return 0

    kept = _select_features(y_full, features_z)
    if not kept:
        logger.warning(
            "models: %s — no features pass %.0f%% coverage, skipping",
            spec.target_name, MIN_FEATURE_COVERAGE * 100,
        )
        return 0

    X_train_full = _aligned_matrix(y_full.index, features_z, kept).dropna()
    y_train = y_full.loc[X_train_full.index]
    if len(X_train_full) < MIN_TRAIN_SAMPLES:
        logger.warning(
            "models: %s — only %d aligned rows (need %d), skipping",
            spec.target_name, len(X_train_full), MIN_TRAIN_SAMPLES,
        )
        return 0

    # ── Walk-forward OOS predictions on realised period ──────────────────────
    oos_dates, oos_preds = _walk_forward_oos(X_train_full, y_train, spec.horizon_days)
    if len(oos_dates) == 0:
        logger.warning(
            "models: %s — walk-forward produced no OOS predictions, skipping",
            spec.target_name,
        )
        return 0

    # ── Final fit on all realised data + diagnostics ─────────────────────────
    final = _new_estimator(spec.horizon_days)
    final.fit(X_train_full.values, y_train.values)

    in_sample_r2 = final.score(X_train_full.values, y_train.values)
    y_oos = y_train.loc[oos_dates].values
    ss_res = float(np.sum((y_oos - oos_preds) ** 2))
    ss_tot = float(np.sum((y_oos - y_oos.mean()) ** 2))
    oos_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    nonzero_coefs = int(np.sum(np.abs(final.coef_) > 1e-10))
    logger.info(
        "models: %s  feats=%d (nz=%d)  n=%d  oos_n=%d  alpha=%.4g  l1=%.2f  "
        "in_R2=%.3f  oos_R2=%.3f",
        spec.target_name, len(kept), nonzero_coefs, len(X_train_full),
        len(oos_dates), final.alpha_, final.l1_ratio_, in_sample_r2, oos_r2,
    )

    # ── Live predictions: dates past the last realised target ────────────────
    # Only require features with NONZERO coefficients to be present — features
    # the model zeroed out contribute nothing regardless of their value, so a
    # missing FRED.GDP shouldn't block a live prediction.
    nonzero_idx = np.where(np.abs(final.coef_) > 1e-10)[0]
    nonzero_feats = [kept[i] for i in nonzero_idx]
    coef_nz = final.coef_[nonzero_idx]

    live_start = y_full.index.max() + pd.Timedelta(days=1)
    live_dates_all = features_z.index[features_z.index >= live_start]
    if len(live_dates_all) == 0:
        live_dates, live_preds = pd.DatetimeIndex([]), np.array([])
    elif len(nonzero_feats) == 0:
        # Constant model — predict the intercept on every live date.
        live_dates = live_dates_all
        live_preds = np.full(len(live_dates), final.intercept_)
    else:
        X_live = _aligned_matrix(live_dates_all, features_z, nonzero_feats).dropna()
        live_dates = X_live.index
        live_preds = X_live.values @ coef_nz + final.intercept_

    # ── Standardise to z-scores using the OOS distribution ───────────────────
    mu = float(np.mean(oos_preds))
    sigma = float(np.std(oos_preds, ddof=0))
    def _z(x: np.ndarray) -> np.ndarray:
        if sigma == 0 or not np.isfinite(sigma):
            return np.full_like(x, np.nan, dtype=float)
        return (x - mu) / sigma

    oos_z = _z(oos_preds)
    live_z = _z(live_preds) if len(live_preds) else np.array([])

    rows: list[dict] = []
    rows.extend(
        {
            "date": d.date(),
            "target_name": spec.target_name,
            "horizon_days": spec.horizon_days,
            "predicted_value": float(p),
            "predicted_z": float(z) if np.isfinite(z) else None,
            "model_version": MODEL_VERSION,
        }
        for d, p, z in zip(oos_dates, oos_preds, oos_z)
    )
    rows.extend(
        {
            "date": d.date(),
            "target_name": spec.target_name,
            "horizon_days": spec.horizon_days,
            "predicted_value": float(p),
            "predicted_z": float(z) if np.isfinite(z) else None,
            "model_version": MODEL_VERSION,
        }
        for d, p, z in zip(live_dates, live_preds, live_z)
    )

    n = _persist_predictions(rows)
    logger.info(
        "models: %s — wrote %d predictions (%d OOS + %d live)",
        spec.target_name, n, len(oos_dates), len(live_dates),
    )
    return n


# ── Public API ───────────────────────────────────────────────────────────────

def run_all() -> dict[str, int]:
    """Train every spec. Returns {target_name: rows_written}."""
    with Session() as session:
        features_z = _load_features_z(session)

    if features_z.empty:
        logger.warning("models: features_z is empty — nothing to do")
        return {}

    summary: dict[str, int] = {}
    for spec in SPECS:
        summary[spec.target_name] = _build_one(spec, features_z)

    total = sum(summary.values())
    logger.info("models: run_all complete — %d predictions across %d targets",
                total, len(SPECS))
    summary["total_rows"] = total
    return summary


if __name__ == "__main__":
    run_all()
