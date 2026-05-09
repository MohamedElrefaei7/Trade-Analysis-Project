"""
builder.py — three-flavour edge-triggered alerter.

(a) feature_extreme:
    A feature's z-score crossed ±FEATURE_Z_THRESHOLD today AND that feature
    has a significant lead-lag relationship (|pearson_r| ≥ SIGNAL_PEARSON_MIN)
    with at least one target in a recent signals snapshot. The "edge" is
    "yesterday's z was inside the threshold (or missing), today's is outside",
    so a feature stuck at z=2.5 for a week fires once, not seven times.

(b) prediction_extreme:
    A model's predicted_z for some (target, horizon) crossed ±PRED_Z_THRESHOLD
    today, with the same edge-trigger semantics applied to the prior day's
    predicted_z for the same (target, horizon, model_version).

(c) regime_change:
    A signals row at as_of_date = today is significant (|pearson_r| ≥
    REGIME_PEARSON_MIN AND granger_p < REGIME_GRANGER_MAX) but the most
    recent prior snapshot for the same (feature, target, window) within
    REGIME_LOOKBACK_DAYS was either absent, weak, or insignificant. This
    catches "this pair's lead-lag relationship just turned on".

All three converge to a list of alert dicts that get upserted into the
`alerts` table (UNIQUE constraint is `NULLS NOT DISTINCT` on Postgres 15+,
so re-runs on the same data are idempotent).

If SLACK_WEBHOOK_URL is set, a single digest message is posted with all
new alerts grouped by severity. Quietly skipped if missing — the DB write
is the source of truth; Slack is optional spice.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from urllib import error as urlerror
from urllib import request as urlrequest

from sqlalchemy import text

from clients.base import Session, logger


# ── Thresholds ───────────────────────────────────────────────────────────────

FEATURE_Z_THRESHOLD = 2.0
PRED_Z_THRESHOLD = 2.0

SIGNAL_PEARSON_MIN = 0.20      # what we'd call a meaningful correlation
SIGNAL_LOOKBACK_DAYS = 7       # how recent a signal must be to count

REGIME_PEARSON_MIN = 0.20
REGIME_GRANGER_MAX = 0.05
REGIME_LOOKBACK_DAYS = 14      # window we compare today's signal against

CRITICAL_Z = 3.0


# ── Severity helper ──────────────────────────────────────────────────────────

def _severity_from_z(z: float) -> str:
    az = abs(z)
    if az >= CRITICAL_Z:
        return "critical"
    if az >= FEATURE_Z_THRESHOLD:
        return "warning"
    return "info"


# ── (a) Feature extremes ─────────────────────────────────────────────────────

_FEATURE_EXTREME_SQL = text(
    """
    WITH today AS (
        SELECT feature_name, z_score
        FROM features
        WHERE date = :d
          AND z_score IS NOT NULL
          AND ABS(z_score) >= :thr
    ),
    yesterday AS (
        SELECT feature_name, z_score
        FROM features
        WHERE date = :prev AND z_score IS NOT NULL
    ),
    -- pick the strongest recent signal per feature
    best_signal AS (
        SELECT DISTINCT ON (feature_name)
            feature_name, target_name, lag_days, window_days,
            pearson_r, granger_p
        FROM signals
        WHERE as_of_date >= :sig_floor
          AND ABS(pearson_r) >= :pmin
        ORDER BY feature_name, ABS(pearson_r) DESC, as_of_date DESC
    )
    SELECT t.feature_name, t.z_score AS z_today,
           y.z_score        AS z_yest,
           s.target_name, s.lag_days, s.window_days,
           s.pearson_r, s.granger_p
    FROM today t
    JOIN best_signal s USING (feature_name)
    LEFT JOIN yesterday y USING (feature_name)
    WHERE y.z_score IS NULL OR ABS(y.z_score) < :thr
    """
)


def _feature_extreme_alerts(session, asof: date) -> list[dict]:
    rows = session.execute(
        _FEATURE_EXTREME_SQL,
        {
            "d": asof,
            "prev": asof - timedelta(days=1),
            "thr": FEATURE_Z_THRESHOLD,
            "sig_floor": asof - timedelta(days=SIGNAL_LOOKBACK_DAYS),
            "pmin": SIGNAL_PEARSON_MIN,
        },
    ).fetchall()

    out: list[dict] = []
    for r in rows:
        z = float(r.z_today)
        prev = "n/a" if r.z_yest is None else f"{float(r.z_yest):+.2f}"
        lead_lag = "leads" if r.lag_days > 0 else ("lags" if r.lag_days < 0 else "contemporaneous with")
        out.append({
            "as_of_date":   asof,
            "alert_type":   "feature_extreme",
            "severity":     _severity_from_z(z),
            "subject":      f"{r.feature_name} crossed z={z:+.2f}",
            "message": (
                f"{r.feature_name} z-score = {z:+.2f} (was {prev} yesterday). "
                f"Best recent signal: {r.feature_name} {lead_lag} "
                f"{r.target_name} by {abs(r.lag_days)}d "
                f"(Pearson={float(r.pearson_r):+.3f}, "
                f"window={r.window_days}d, "
                f"granger_p={'n/a' if r.granger_p is None else f'{float(r.granger_p):.3f}'})."
            ),
            "feature_name": r.feature_name,
            "target_name":  r.target_name,
            "horizon_days": None,
            "window_days":  int(r.window_days),
            "z_score":      z,
            "pearson_r":    float(r.pearson_r),
        })
    return out


# ── (b) Prediction extremes ──────────────────────────────────────────────────

_PRED_EXTREME_SQL = text(
    """
    SELECT t.target_name, t.horizon_days, t.model_version,
           t.predicted_value, t.predicted_z AS z_today,
           y.predicted_z                   AS z_yest
    FROM predictions t
    LEFT JOIN predictions y
        ON  y.target_name   = t.target_name
        AND y.horizon_days  = t.horizon_days
        AND y.model_version = t.model_version
        AND y.date          = t.date - INTERVAL '1 day'
    WHERE t.date = :d
      AND t.predicted_z IS NOT NULL
      AND ABS(t.predicted_z) >= :thr
      AND (y.predicted_z IS NULL OR ABS(y.predicted_z) < :thr)
    """
)


def _prediction_extreme_alerts(session, asof: date) -> list[dict]:
    rows = session.execute(
        _PRED_EXTREME_SQL, {"d": asof, "thr": PRED_Z_THRESHOLD}
    ).fetchall()

    out: list[dict] = []
    for r in rows:
        z = float(r.z_today)
        prev = "n/a" if r.z_yest is None else f"{float(r.z_yest):+.2f}"
        direction = "bullish" if z > 0 else "bearish"
        out.append({
            "as_of_date":   asof,
            "alert_type":   "prediction_extreme",
            "severity":     _severity_from_z(z),
            "subject":      f"{r.target_name} prediction z={z:+.2f}",
            "message": (
                f"Model {r.model_version} predicts {r.target_name} "
                f"({r.horizon_days}d horizon) = {float(r.predicted_value):+.4f}, "
                f"z={z:+.2f} ({direction}). "
                f"Yesterday z={prev}."
            ),
            "feature_name": None,
            "target_name":  r.target_name,
            "horizon_days": int(r.horizon_days),
            "window_days":  None,
            "z_score":      z,
            "pearson_r":    None,
        })
    return out


# ── (c) Regime change in signals ─────────────────────────────────────────────

_REGIME_SQL = text(
    """
    WITH today_sig AS (
        SELECT feature_name, target_name, window_days, lag_days,
               pearson_r, granger_p
        FROM signals
        WHERE as_of_date = :d
          AND ABS(pearson_r) >= :pmin
          AND granger_p IS NOT NULL
          AND granger_p < :gmax
    ),
    prev_sig AS (
        SELECT DISTINCT ON (feature_name, target_name, window_days)
            feature_name, target_name, window_days,
            pearson_r, granger_p, as_of_date
        FROM signals
        WHERE as_of_date <  :d
          AND as_of_date >= :prev_floor
        ORDER BY feature_name, target_name, window_days, as_of_date DESC
    )
    SELECT t.feature_name, t.target_name, t.window_days, t.lag_days,
           t.pearson_r, t.granger_p,
           p.pearson_r AS prev_pearson, p.granger_p AS prev_granger,
           p.as_of_date AS prev_as_of
    FROM today_sig t
    LEFT JOIN prev_sig p USING (feature_name, target_name, window_days)
    WHERE p.feature_name IS NULL
       OR ABS(p.pearson_r) < :pmin
       OR p.granger_p IS NULL
       OR p.granger_p >= :gmax
    """
)


def _regime_change_alerts(session, asof: date) -> list[dict]:
    rows = session.execute(
        _REGIME_SQL,
        {
            "d": asof,
            "prev_floor": asof - timedelta(days=REGIME_LOOKBACK_DAYS),
            "pmin": REGIME_PEARSON_MIN,
            "gmax": REGIME_GRANGER_MAX,
        },
    ).fetchall()

    out: list[dict] = []
    for r in rows:
        pr = float(r.pearson_r)
        gp = float(r.granger_p)
        if r.prev_pearson is None:
            prior = "no prior snapshot in lookback window"
        else:
            prior = (
                f"prior {r.prev_as_of}: "
                f"Pearson={float(r.prev_pearson):+.3f}, "
                f"granger_p="
                f"{'n/a' if r.prev_granger is None else f'{float(r.prev_granger):.3f}'}"
            )
        lead_lag = "leads" if r.lag_days > 0 else ("lags" if r.lag_days < 0 else "contemporaneous with")
        out.append({
            "as_of_date":   asof,
            "alert_type":   "regime_change",
            "severity":     "info",
            "subject":      f"new signal: {r.feature_name} → {r.target_name}",
            "message": (
                f"{r.feature_name} {lead_lag} {r.target_name} by "
                f"{abs(r.lag_days)}d (window={r.window_days}d): "
                f"Pearson={pr:+.3f}, granger_p={gp:.3f}. "
                f"Just became significant — {prior}."
            ),
            "feature_name": r.feature_name,
            "target_name":  r.target_name,
            "horizon_days": None,
            "window_days":  int(r.window_days),
            "z_score":      None,
            "pearson_r":    pr,
        })
    return out


# ── Persistence ──────────────────────────────────────────────────────────────

_UPSERT_SQL = text(
    """
    INSERT INTO alerts
        (as_of_date, alert_type, severity, subject, message,
         feature_name, target_name, horizon_days, window_days,
         z_score, pearson_r)
    VALUES
        (:as_of_date, :alert_type, :severity, :subject, :message,
         :feature_name, :target_name, :horizon_days, :window_days,
         :z_score, :pearson_r)
    ON CONFLICT
        (as_of_date, alert_type, feature_name, target_name, horizon_days, window_days)
    DO UPDATE SET
        severity     = EXCLUDED.severity,
        subject      = EXCLUDED.subject,
        message      = EXCLUDED.message,
        z_score      = EXCLUDED.z_score,
        pearson_r    = EXCLUDED.pearson_r,
        triggered_at = NOW()
    RETURNING (xmax = 0) AS inserted
    """
)


def _persist(rows: list[dict]) -> tuple[int, int]:
    """Returns (inserted, upserted_total)."""
    if not rows:
        return (0, 0)
    inserted = 0
    with Session() as s:
        for row in rows:
            res = s.execute(_UPSERT_SQL, row).fetchone()
            if res and res.inserted:
                inserted += 1
        s.commit()
    return (inserted, len(rows))


# ── Slack digest (optional) ──────────────────────────────────────────────────

def _maybe_post_slack(new_alerts: list[dict], asof: date) -> bool:
    """Post a digest to Slack if SLACK_WEBHOOK_URL is set. Returns True on send."""
    if not new_alerts:
        return False
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False

    by_sev: dict[str, list[dict]] = {"critical": [], "warning": [], "info": []}
    for a in new_alerts:
        by_sev.setdefault(a["severity"], []).append(a)

    lines = [f"*Alerter — {asof.isoformat()} ({len(new_alerts)} new)*"]
    for sev in ("critical", "warning", "info"):
        items = by_sev.get(sev) or []
        if not items:
            continue
        emoji = {"critical": ":rotating_light:", "warning": ":warning:", "info": ":information_source:"}[sev]
        lines.append(f"\n{emoji} *{sev.upper()}* ({len(items)})")
        for a in items[:10]:
            lines.append(f"  • [{a['alert_type']}] {a['subject']}")
        if len(items) > 10:
            lines.append(f"  …and {len(items) - 10} more")

    payload = json.dumps({"text": "\n".join(lines)}).encode("utf-8")
    req = urlrequest.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                logger.warning("alerts: Slack returned HTTP %s", resp.status)
            return ok
    except urlerror.URLError as exc:
        logger.warning("alerts: Slack post failed (%s)", exc)
        return False


# ── Public API ───────────────────────────────────────────────────────────────

def _resolve_asof(session) -> date | None:
    """Use the latest features.date as the audit anchor — features.z_score is
    what (a) and (c) hinge on, so this is the date both checks line up on."""
    row = session.execute(
        text("SELECT MAX(date) FROM features WHERE z_score IS NOT NULL")
    ).fetchone()
    return row[0] if row and row[0] else None


def build(asof: date | None = None) -> dict[str, int]:
    """
    Run all three checks for `asof` (defaults to latest features date).
    Returns counts: {alert_type: rows_inserted}.
    """
    with Session() as session:
        if asof is None:
            asof = _resolve_asof(session)
            if asof is None:
                logger.warning("alerts: features table is empty — nothing to do")
                return {}

        feature_rows = _feature_extreme_alerts(session, asof)
        pred_rows    = _prediction_extreme_alerts(session, asof)
        regime_rows  = _regime_change_alerts(session, asof)

    all_rows = feature_rows + pred_rows + regime_rows
    inserted, total = _persist(all_rows)

    summary = {
        "feature_extreme":    len(feature_rows),
        "prediction_extreme": len(pred_rows),
        "regime_change":      len(regime_rows),
        "total_candidates":   total,
        "newly_inserted":     inserted,
    }
    logger.info(
        "alerts: as_of=%s  feature_extreme=%d  prediction_extreme=%d  "
        "regime_change=%d  new=%d (of %d)",
        asof, len(feature_rows), len(pred_rows), len(regime_rows),
        inserted, total,
    )

    # Only digest the freshly-inserted rows so Slack doesn't re-spam
    # alerts that already fired on a previous run.
    if inserted:
        new_only = [r for r in all_rows if _is_new(r, asof)]
        # Best-effort: if we can't perfectly identify new rows, fall back to all.
        digest = new_only if new_only else all_rows
        if _maybe_post_slack(digest, asof):
            logger.info("alerts: posted Slack digest (%d items)", len(digest))

    return summary


def _is_new(row: dict, asof: date) -> bool:
    """Check whether this exact alert row was inserted (vs updated) just now.
    A fresh row will have triggered_at within the last few seconds."""
    sql = text(
        """
        SELECT EXTRACT(EPOCH FROM NOW() - triggered_at) AS age_sec
        FROM alerts
        WHERE as_of_date  = :as_of_date
          AND alert_type  = :alert_type
          AND feature_name IS NOT DISTINCT FROM :feature_name
          AND target_name  IS NOT DISTINCT FROM :target_name
          AND horizon_days IS NOT DISTINCT FROM :horizon_days
          AND window_days  IS NOT DISTINCT FROM :window_days
        """
    )
    with Session() as s:
        r = s.execute(sql, row).fetchone()
    return bool(r and r.age_sec is not None and r.age_sec < 60)


def run_all() -> dict[str, int]:
    """Entry point for the nightly Prefect flow."""
    return build()


if __name__ == "__main__":
    run_all()
