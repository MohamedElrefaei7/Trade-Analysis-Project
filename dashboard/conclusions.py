"""
conclusions.py
==============

Turns rows from the `features`, `signals`, and `predictions` tables into a
prioritized list of human-readable conclusions for the Today tab.

DESIGN PHILOSOPHY
-----------------
This module exists because the dashboard's job is to answer "what should I pay
attention to right now?" — not to display data. Charts are evidence; the
conclusion is the claim. If we can't write the claim as a sentence, it doesn't
belong on the landing page.

The contract is deliberately narrow:
    - Input: dataframes pulled from the four tables, plus a config object.
    - Output: list[Conclusion], already sorted by severity, capped at N.
The rendering layer (streamlit_app.py) iterates and formats. No SQL, no
plotting, no Streamlit imports here — that separation is what makes this
testable. You should be able to feed synthetic dataframes into
generate_conclusions() in a unit test and assert on the output structure.

SCHEMA NOTES (matches schema.sql)
---------------------------------
- `signals` uses `pearson_r` (NOT `correlation`) and `as_of_date` (NOT `date`).
- `predictions` carries `predicted_value` AND `predicted_z` (z is computed
  against the OOS prediction distribution by the trainer; we use it directly
  rather than recomputing percentiles every render).
- `features` rows have one (date, feature_name) per row with `value` and
  `z_score`. "features_today" below is the slice WHERE date = max(date).

WHY A SCORE-BASED RANKING INSTEAD OF SEPARATE LISTS PER TYPE
------------------------------------------------------------
Earlier drafts had three separate functions returning three separate lists
(threshold breaches, regime changes, model extremes). That made the Today tab
feel cluttered: you'd see a yellow regime-change conclusion above a red
threshold breach because they were in different sections. The user has to do
the prioritization mentally. Bad.

Instead, every conclusion type computes a unified `severity_score` on the same
0-100 scale, and we sort across all types together. A 2.5σ feature breach with
a strong historical signal outranks a marginal regime change every time,
regardless of which detector found it. This is the single most important
design choice in the file.

WHY WE CAP AT 5
---------------
Empirically, anything past the 5th conclusion is noise the user will scroll
past. Capping forces the scoring function to be honest — if you're tempted to
raise the cap, the real fix is usually to tighten the thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _z_label(z: float) -> str:
    az = abs(z)
    if az >= 2.5:
        return "unusually high" if z > 0 else "unusually low"
    if az >= 2.0:
        return "elevated" if z > 0 else "depressed"
    if az >= 1.5:
        return "slightly elevated" if z > 0 else "slightly depressed"
    return "near normal"


def _format_headline(conclusion_type: str, **kwargs) -> str:
    """Single source of truth for all conclusion headline templates."""
    if conclusion_type == "threshold_breach":
        return (
            f"Higher probability of {kwargs['direction']} {kwargs['target']} "
            f"over the next {kwargs['horizon']}d. "
            f"{kwargs['feature']} is {kwargs['z_label']}, "
            f"and this has historically preceded {kwargs['target']} moves "
            f"with r={kwargs['r']:+.2f}."
        )
    if conclusion_type == "regime_change":
        framing = kwargs["framing"]
        if framing == "newly significant":
            return (
                f"New signal forming: {kwargs['feature']} now tracks "
                f"{kwargs['target']} (r went from {kwargs['prior']:+.2f} to "
                f"{kwargs['recent']:+.2f} over the last {kwargs['days']} days)."
            )
        if framing == "broke down":
            return (
                f"Signal weakening: {kwargs['feature']} → {kwargs['target']} "
                f"relationship has faded (r dropped from {kwargs['prior']:+.2f} "
                f"to {kwargs['recent']:+.2f})."
            )
        return (
            f"{kwargs['feature']} → {kwargs['target']} correlation shifted "
            f"from {kwargs['prior']:+.2f} to {kwargs['recent']:+.2f} "
            f"over the last {kwargs['days']} days."
        )
    if conclusion_type == "model_extreme":
        return (
            f"Model leaning {kwargs['direction']} on {kwargs['target']}: "
            f"prediction is in the {kwargs['pct_ordinal']} percentile of its history."
        )
    if conclusion_type == "stale_data":
        return (
            f"{kwargs['n']} feature(s) haven't updated recently — "
            f"treat related conclusions with caution."
        )
    if conclusion_type == "no_signal":
        return "Nothing notable today. All features within normal range."
    return ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Severity = Literal["critical", "warning", "info"]
ConclusionType = Literal[
    "threshold_breach",
    "regime_change",
    "model_extreme",
    "stale_data",
    "no_signal",
]


@dataclass
class Conclusion:
    """
    A single rendered conclusion. The rendering layer reads these fields
    directly — don't add presentation logic here, but DO include all the
    numbers the headline references so the renderer doesn't have to re-derive
    them.

    `evidence_link` is a dict rather than a URL because Streamlit doesn't have
    real URLs for tab navigation; the renderer translates this dict into a
    session_state mutation that switches tabs and pre-applies filters. Keeping
    it structured (not a URL string) means the renderer can change navigation
    mechanics without touching this module.
    """
    severity: Severity
    severity_score: float                # 0-100, used for sorting
    conclusion_type: ConclusionType
    headline: str                        # The sentence shown to the user
    supporting_numbers: dict             # {key: value} for the renderer
    evidence_link: Optional[dict] = None  # {"tab": "Signals", "filters": {...}}
    as_of: date = field(default_factory=date.today)


@dataclass
class ConclusionConfig:
    """
    All thresholds in one place so they're tunable from the dashboard sidebar
    without code changes. Defaults below are starting points — expect to tune
    these once you have a few weeks of dashboard use and a sense of how often
    each conclusion type fires.

    NOTE on z-score thresholds: 2.0σ is the textbook "significant" line, but
    with N features checked daily you'll see roughly one 2σ breach every
    couple of days by chance alone. We use 2.0 as the warning floor and 2.5
    as the critical floor to keep false-positive rate manageable. Adjust
    after you've measured your actual false-positive rate against forward
    target moves.
    """
    # Feature breach thresholds
    z_warning: float = 2.0
    z_critical: float = 2.5

    # Signal quality gates — a breach is only "actionable" if the feature has
    # a historically meaningful relationship with at least one target. Without
    # this gate, every random spike in a noisy series gets surfaced.
    min_abs_correlation: float = 0.35
    max_granger_p: float = 0.10
    min_sample_size: int = 100

    # Regime change: how much the rolling correlation has to have shifted
    # for us to flag it. 0.20 is roughly "the relationship strength has
    # noticeably changed" — smaller deltas are within noise.
    regime_change_threshold: float = 0.20
    regime_change_lookback_days: int = 30

    # Model extreme: predictions in the top/bottom percentile of their own
    # historical distribution. We use percentiles instead of σ because model
    # outputs aren't necessarily normal. The trainer also writes `predicted_z`
    # against the OOS distribution; either anchor works — percentiles are
    # more robust to fat tails.
    model_extreme_percentile: float = 0.90  # top/bottom 10%

    # Data freshness — anything older than this gets a stale_data conclusion
    # rather than being silently used.
    max_feature_age_days: int = 3

    # Hard cap on conclusions surfaced.
    max_conclusions: int = 5


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------
#
# All conclusion types funnel into score_severity() so the ranking is
# comparable across types. The function is intentionally simple: a base score
# per type, scaled by magnitude, with a multiplier for signal quality.
#
# Why these specific weights:
#   - Threshold breaches with strong historical signal are the most
#     actionable thing this system produces, so they top out at 100.
#   - Regime changes are interesting but slower-moving — they cap at ~70,
#     because acting on a regime change requires more deliberation than
#     acting on a breach.
#   - Model extremes cap at ~85 because they're a synthesis of multiple
#     features, but their interpretability is lower than a single feature
#     breach (the user has to trust the model).
#   - Stale data is a 60 — it's important enough to surface above noise,
#     but not so important it should outrank actual signals.
#
# Tune these by gut feel after a few weeks of use; there's no principled
# answer.

def _score_threshold_breach(z: float, abs_r: float, granger_p: float | None) -> float:
    z_component = min(50 + (abs(z) - 2.0) * 25, 100)
    # Granger p may legitimately be NULL when the trainer didn't run it for
    # this row (e.g. statsmodels disabled). Treat as the worst tolerated p
    # so we don't reward missing data.
    gp = 0.10 if granger_p is None else granger_p
    quality = (abs_r / 0.7) * (1 - gp / 0.10)
    quality = float(np.clip(quality, 0.5, 1.0))
    return z_component * quality


def _score_regime_change(correlation_delta: float, current_abs_r: float) -> float:
    base = min(40 + abs(correlation_delta) * 100, 70)
    return base * (current_abs_r / 0.5)


def _score_model_extreme(percentile: float) -> float:
    distance_from_median = abs(percentile - 0.5) * 2  # 0 to 1
    return 40 + distance_from_median * 45


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------
#
# Each detector returns list[Conclusion]. They're independent and side-effect
# free. The orchestrator below calls them all and merges the output.
#
# Convention: detectors NEVER filter their own output by severity_score. The
# ranking and capping happens once, at the end, in generate_conclusions().
# This keeps the logic for "which conclusions exist" separate from "which
# ones do we show." If a future caller wants the full list (e.g., for an
# email digest), they can call generate_conclusions(cap=None).


def detect_threshold_breaches(
    features_today: pd.DataFrame,
    signals: pd.DataFrame,
    config: ConclusionConfig,
) -> list[Conclusion]:
    """
    A feature is breaching when |z| >= z_warning AND it has at least one
    qualifying historical signal against any target.

    The signal-quality gate is critical: the features table contains plenty
    of series that occasionally spike for reasons unrelated to anything we
    care about (FX noise, data revisions, holiday effects). Without the gate,
    those spikes drown out real signals.

    For each breaching feature, we surface ONE conclusion using its strongest
    historical signal (highest |r|), not one conclusion per (feature, target)
    pair. Otherwise a single feature with strong relationships to three
    targets would consume three of the five conclusion slots.
    """
    conclusions: list[Conclusion] = []
    if features_today.empty or "z_score" not in features_today.columns:
        return conclusions

    breaching = features_today[
        features_today["z_score"].abs() >= config.z_warning
    ]

    for _, feat_row in breaching.iterrows():
        feature_name = feat_row["feature_name"]
        z = float(feat_row["z_score"])

        feat_signals = signals[
            (signals["feature_name"] == feature_name)
            & (signals["pearson_r"].abs() >= config.min_abs_correlation)
            & (
                signals["granger_p"].isna()
                | (signals["granger_p"] <= config.max_granger_p)
            )
            & (signals["sample_size"] >= config.min_sample_size)
        ]
        if feat_signals.empty:
            # Feature is breaching but has no actionable historical
            # relationship — don't surface as a conclusion. Better to stay
            # silent than cry wolf.
            continue

        best = feat_signals.loc[feat_signals["pearson_r"].abs().idxmax()]
        abs_r = float(abs(best["pearson_r"]))
        granger_p = (
            None if pd.isna(best["granger_p"]) else float(best["granger_p"])
        )
        score = _score_threshold_breach(z, abs_r, granger_p)

        # Direction logic: if z is positive and correlation is positive, the
        # implication is bullish for the target (and vice versa). This is the
        # whole reason we built the signals table — we can translate a
        # feature move into a directional claim about a target.
        target_direction = (
            "higher" if (z > 0) == (best["pearson_r"] > 0) else "lower"
        )

        severity: Severity = (
            "critical" if abs(z) >= config.z_critical else "warning"
        )

        lag_days = int(best["lag_days"])

        headline = _format_headline(
            "threshold_breach",
            direction=target_direction,
            target=best["target_name"],
            horizon=max(abs(lag_days), 1),
            feature=feature_name,
            z_label=_z_label(z),
            r=float(best["pearson_r"]),
        )

        conclusions.append(Conclusion(
            severity=severity,
            severity_score=score,
            conclusion_type="threshold_breach",
            headline=headline,
            supporting_numbers={
                "feature_name": feature_name,
                "target_name": best["target_name"],
                "z_score": z,
                "feature_value": float(feat_row.get("value", float("nan"))),
                "pearson_r": float(best["pearson_r"]),
                "lag_days": lag_days,
                "window_days": int(best["window_days"]),
                "sample_size": int(best["sample_size"]),
                "granger_p": granger_p,
                "implication_direction": target_direction,
            },
            evidence_link={
                "tab": "Signals",
                "filters": {
                    "feature_name": feature_name,
                    "target_name": best["target_name"],
                },
            },
        ))

    return conclusions


def detect_regime_changes(
    signals_history: pd.DataFrame,
    config: ConclusionConfig,
) -> list[Conclusion]:
    """
    A regime change is when the rolling correlation between a (feature, target,
    window) triple has shifted materially in the recent window vs. the prior
    window.

    Expected input: signals table with multiple `as_of_date` rows per
    (feature, target, window) — i.e., the nightly recompute writes a new row
    each day rather than upserting. If your schema upserts (single row per
    pair), this detector won't work and you should change the schema; the
    historical correlation trajectory is too valuable to throw away.

    Why this matters: a relationship that just became significant is much
    more interesting than one that's been significant for years. The first
    case is news; the second is already priced into your worldview.
    """
    conclusions: list[Conclusion] = []
    if signals_history.empty:
        return conclusions

    today = pd.Timestamp(signals_history["as_of_date"].max())
    recent_cutoff = today - pd.Timedelta(days=config.regime_change_lookback_days)

    grouped = signals_history.groupby(
        ["feature_name", "target_name", "window_days"]
    )
    for (feature, target, window), group in grouped:
        recent = group[
            pd.to_datetime(group["as_of_date"]) >= recent_cutoff
        ]["pearson_r"]
        prior = group[
            pd.to_datetime(group["as_of_date"]) < recent_cutoff
        ]["pearson_r"]
        if len(recent) < 5 or len(prior) < 30:
            # Not enough history on either side for a meaningful comparison.
            # Bias toward caution — we'd rather miss a real regime change
            # than fire on a sparse one.
            continue

        delta = float(recent.mean() - prior.mean())
        if abs(delta) < config.regime_change_threshold:
            continue

        current_abs_r = float(abs(recent.mean()))
        prior_abs_r = float(abs(prior.mean()))
        score = _score_regime_change(delta, current_abs_r)

        if prior_abs_r < config.min_abs_correlation <= current_abs_r:
            framing = "newly significant"
            sev: Severity = "warning"
        elif current_abs_r < config.min_abs_correlation <= prior_abs_r:
            framing = "broke down"
            sev = "info"
        else:
            framing = "shifted"
            sev = "info"

        latest_lag = group.sort_values("as_of_date").iloc[-1]["lag_days"]
        headline = _format_headline(
            "regime_change",
            framing=framing,
            feature=feature,
            target=target,
            prior=float(prior.mean()),
            recent=float(recent.mean()),
            days=config.regime_change_lookback_days,
        )

        conclusions.append(Conclusion(
            severity=sev,
            severity_score=score,
            conclusion_type="regime_change",
            headline=headline,
            supporting_numbers={
                "feature_name": feature,
                "target_name": target,
                "window_days": int(window),
                "prior_correlation": float(prior.mean()),
                "recent_correlation": float(recent.mean()),
                "delta": delta,
                "lag_days": int(latest_lag),
                "framing": framing,
            },
            evidence_link={
                "tab": "Signals",
                "filters": {
                    "feature_name": feature,
                    "target_name": target,
                    "show_stability": True,
                },
            },
        ))

    return conclusions


def detect_model_extremes(
    predictions_today: pd.DataFrame,
    predictions_history: pd.DataFrame,
    config: ConclusionConfig,
) -> list[Conclusion]:
    """
    A model prediction in the top/bottom decile of its own historical
    distribution is a synthesis-level signal: the model thinks today's
    feature configuration is unusually predictive in one direction.

    We compare against the model's own history rather than against a
    Gaussian assumption because ElasticNet/LightGBM outputs aren't
    distributed nicely, especially during regime shifts.

    NOTE: this detector silently skips any (target, horizon, model_version)
    triple with fewer than 60 historical predictions — the percentile claim
    is meaningless on a thin sample.
    """
    conclusions: list[Conclusion] = []
    if predictions_today.empty:
        return conclusions

    for _, pred_row in predictions_today.iterrows():
        target = pred_row["target_name"]
        horizon = int(pred_row["horizon_days"])
        model_version = pred_row.get("model_version", None)
        prediction = float(pred_row["predicted_value"])

        hist = predictions_history[
            (predictions_history["target_name"] == target)
            & (predictions_history["horizon_days"] == horizon)
        ]
        if model_version is not None and "model_version" in hist.columns:
            hist = hist[hist["model_version"] == model_version]
        hist_vals = hist["predicted_value"].dropna()

        if len(hist_vals) < 60:
            continue

        percentile = float((hist_vals < prediction).mean())
        # Only fire if extreme on either tail.
        if (
            percentile < config.model_extreme_percentile
            and percentile > (1 - config.model_extreme_percentile)
        ):
            continue

        score = _score_model_extreme(percentile)
        direction = "bullish" if prediction > hist_vals.median() else "bearish"

        headline = _format_headline(
            "model_extreme",
            direction=direction,
            target=target,
            pct_ordinal=_ordinal(int(percentile * 100)),
        )

        conclusions.append(Conclusion(
            severity="warning",
            severity_score=score,
            conclusion_type="model_extreme",
            headline=headline,
            supporting_numbers={
                "target_name": target,
                "horizon_days": horizon,
                "predicted_value": prediction,
                "predicted_z": (
                    None
                    if pd.isna(pred_row.get("predicted_z"))
                    else float(pred_row["predicted_z"])
                ),
                "percentile": percentile,
                "historical_median": float(hist_vals.median()),
                "model_version": model_version,
            },
            evidence_link={
                "tab": "Predictions",
                "filters": {"target_name": target, "horizon_days": horizon},
            },
        ))

    return conclusions


def detect_stale_data(
    features_today: pd.DataFrame,
    config: ConclusionConfig,
) -> list[Conclusion]:
    """
    Surfaces features whose latest value is older than max_feature_age_days.
    We collapse all stale features into a SINGLE conclusion rather than one
    per feature — a list of seven stale features is a single operational
    issue, not seven separate things to think about.
    """
    if features_today.empty or "date" not in features_today.columns:
        return []

    today = pd.Timestamp(date.today())
    df = features_today.copy()
    df["age_days"] = (today - pd.to_datetime(df["date"])).dt.days
    stale = df[df["age_days"] > config.max_feature_age_days]
    if stale.empty:
        return []

    stale_names = stale["feature_name"].tolist()
    headline = _format_headline("stale_data", n=len(stale_names))

    return [Conclusion(
        severity="warning",
        severity_score=60.0,
        conclusion_type="stale_data",
        headline=headline,
        supporting_numbers={
            "stale_count": len(stale_names),
            "stale_features": stale_names,
        },
        evidence_link={"tab": "Health"},
    )]


# ---------------------------------------------------------------------------
# Stability indicator (used by the Signals tab, not the conclusions list).
# Lives here because it's a property of a (feature, target) pair derivable
# from signals_history, and we want it next to the regime-change detector
# that consumes the same input.
# ---------------------------------------------------------------------------

def signal_stability(
    signals_history: pd.DataFrame,
    feature_name: str,
    target_name: str,
    n_windows: int = 3,
) -> list[float]:
    """
    Return the mean Pearson r computed over `n_windows` non-overlapping
    sub-windows of the available history for this (feature, target) pair.

    A signal whose three sub-window correlations are (0.6, 0.6, 0.65) is real;
    one with (0.8, 0.1, 0.5) is a noise average. The Signals tab plots these
    three numbers as dots on a number line — that's the single feature most
    likely to keep us from acting on spurious signals.
    """
    pair = signals_history[
        (signals_history["feature_name"] == feature_name)
        & (signals_history["target_name"] == target_name)
    ].sort_values("as_of_date")
    if len(pair) < n_windows * 5:
        return []

    series = pair["pearson_r"].values
    chunks = np.array_split(series, n_windows)
    results = [float(c.mean()) for c in chunks if len(c) > 0]
    return [v for v in results if not np.isnan(v)]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_conclusions(
    features_today: pd.DataFrame,
    signals: pd.DataFrame,
    signals_history: pd.DataFrame,
    predictions_today: pd.DataFrame,
    predictions_history: pd.DataFrame,
    config: Optional[ConclusionConfig] = None,
) -> list[Conclusion]:
    """
    The single entry point the Streamlit app calls. Runs every detector,
    merges, ranks, caps.

    The cap is applied AFTER ranking, never before. A noisy detector that
    produces 50 low-score conclusions can't crowd out a high-score conclusion
    from a quiet detector — they all compete on severity_score.

    If no detectors fire, we return a single "no_signal" conclusion rather
    than an empty list. An empty Today tab is ambiguous ("is the system
    broken?"); an explicit "nothing notable today" sentence is reassuring.
    The renderer should style this differently from real conclusions.
    """
    config = config or ConclusionConfig()

    all_conclusions = (
        detect_threshold_breaches(features_today, signals, config)
        + detect_regime_changes(signals_history, config)
        + detect_model_extremes(predictions_today, predictions_history, config)
        + detect_stale_data(features_today, config)
    )

    if not all_conclusions:
        return [Conclusion(
            severity="info",
            severity_score=0.0,
            conclusion_type="no_signal",
            headline=_format_headline("no_signal"),
            supporting_numbers={},
        )]

    all_conclusions.sort(key=lambda c: c.severity_score, reverse=True)
    return all_conclusions[: config.max_conclusions]
