"""
streamlit_app.py — the daily monitoring dashboard.

Five tabs, in priority order:

  1. Today        — five sentence-form conclusions, target outlook strip,
                    "what changed since yesterday" delta table.
  2. Signals      — filterable lead-lag grid, on-select evidence plots,
                    stability indicator across three sub-windows.
  3. Predictions  — per-target model trace, residual MAE band, model-health
                    badge, today's per-feature contribution bars.
  4. Explore      — the legacy correlation tool, kept as-is for ad-hoc work.
  5. Health       — pipeline freshness, alert flow, model retraining audit.

Run:
    streamlit run dashboard/streamlit_app.py

The Today tab is the only thing that matters on a normal day — every other
tab exists to justify or interrogate something that surfaced there. Charts
are evidence; the sentence is the claim.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_GRANGER = True
except Exception:
    _HAS_GRANGER = False

# Streamlit runs this file as a script, so neither `dashboard.` nor
# `clients.` is on sys.path. Make both work whether you launch from the
# project root or from inside dashboard/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from clients.base import engine
from dashboard.conclusions import (
    Conclusion,
    ConclusionConfig,
    _ordinal,
    generate_conclusions,
    signal_stability,
)


# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Analysis — Daily Monitor",
    layout="wide",
)


# ── Data access ───────────────────────────────────────────────────────────────
def get_engine():
    """Return the shared SQLAlchemy engine from clients.base."""
    return engine


@st.cache_data(ttl=3600)
def load_features_latest() -> pd.DataFrame:
    """Latest row per feature_name. Today tab + cards live on this slice."""
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT DISTINCT ON (feature_name)
                    feature_name, date, value, z_score
                FROM features
                WHERE z_score IS NOT NULL
                ORDER BY feature_name, date DESC
                """
            ),
            conn,
        )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


@st.cache_data(ttl=3600)
def load_features_for_delta(days_back: int = 30) -> pd.DataFrame:
    """Recent feature rows so we can compute today vs. yesterday deltas."""
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT date, feature_name, value, z_score
                FROM features
                WHERE date >= :floor
                  AND z_score IS NOT NULL
                ORDER BY date
                """
            ),
            conn,
            params={"floor": date.today() - timedelta(days=days_back)},
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_features_history(start: date, end: date) -> pd.DataFrame:
    """Wide-form (date × feature) for plotting + the Explore tab."""
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT date, feature_name, value, z_score
                FROM features
                WHERE date BETWEEN :s AND :e
                ORDER BY date
                """
            ),
            conn,
            params={"s": start, "e": end},
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def list_features() -> list[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT feature_name FROM features ORDER BY feature_name")
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=3600)
def list_targets() -> list[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT target_name FROM targets ORDER BY target_name")
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=3600)
def load_target_history(target_name: str) -> pd.DataFrame:
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT date, value, horizon_days
                FROM targets
                WHERE target_name = :t
                  AND value IS NOT NULL
                ORDER BY date
                """
            ),
            conn,
            params={"t": target_name},
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_signals_latest() -> pd.DataFrame:
    """One row per (feature, target, window) — the most recent snapshot."""
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT DISTINCT ON (feature_name, target_name, window_days)
                    as_of_date, feature_name, target_name, window_days,
                    lag_days, pearson_r, spearman_r, granger_p, sample_size
                FROM signals
                ORDER BY feature_name, target_name, window_days, as_of_date DESC
                """
            ),
            conn,
        )
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    return df


@st.cache_data(ttl=3600)
def load_signals_history(days_back: int = 365) -> pd.DataFrame:
    """All signals snapshots for the regime-change detector + stability dots."""
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT as_of_date, feature_name, target_name, window_days,
                       lag_days, pearson_r, spearman_r, granger_p, sample_size
                FROM signals
                WHERE as_of_date >= :floor
                """
            ),
            conn,
            params={"floor": date.today() - timedelta(days=days_back)},
        )
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    return df


@st.cache_data(ttl=3600)
def load_predictions_latest() -> pd.DataFrame:
    """The most recent prediction per (target, horizon, model_version)."""
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT DISTINCT ON (target_name, horizon_days, model_version)
                    date, target_name, horizon_days, predicted_value,
                    predicted_z, model_version
                FROM predictions
                ORDER BY target_name, horizon_days, model_version, date DESC
                """
            ),
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_predictions_history(target_name: str | None = None) -> pd.DataFrame:
    sql = """
        SELECT date, target_name, horizon_days, predicted_value,
               predicted_z, model_version
        FROM predictions
        {where}
        ORDER BY date
    """
    where = "WHERE target_name = :t" if target_name else ""
    params = {"t": target_name} if target_name else {}
    with get_engine().connect() as conn:
        df = pd.read_sql(text(sql.format(where=where)), conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_alerts_recent(days: int = 14) -> pd.DataFrame:
    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT triggered_at, as_of_date, alert_type, severity,
                       subject, message, feature_name, target_name,
                       horizon_days, window_days, z_score, pearson_r
                FROM alerts
                WHERE as_of_date >= :floor
                ORDER BY triggered_at DESC
                """
            ),
            conn,
            params={"floor": date.today() - timedelta(days=days)},
        )
    if not df.empty:
        df["triggered_at"] = pd.to_datetime(df["triggered_at"])
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    return df


# ── Session-state helpers ─────────────────────────────────────────────────────
#
# Streamlit resets widget state on every rerun unless we route it through
# st.session_state. We use it for two things:
#   1. Cross-tab navigation: clicking a Today conclusion sets `nav_to_tab`
#      and `nav_filters`, which the target tab then consumes as defaults.
#   2. Filter persistence on the Signals tab, so flipping between tabs
#      doesn't blow away your min-r threshold.

def _ss(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _navigate(link: dict | None) -> None:
    """Translate an evidence_link dict into session_state mutations the
    consuming tab will pick up. Call this from a button on_click handler."""
    if not link:
        return
    st.session_state["nav_to_tab"] = link.get("tab")
    st.session_state["nav_filters"] = link.get("filters", {})


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Global filters")

today = pd.Timestamp(date.today())
default_start = (today - pd.Timedelta(days=365)).date()

if "global_start" not in st.session_state:
    st.session_state["global_start"] = default_start
if "global_end" not in st.session_state:
    st.session_state["global_end"] = today.date()

st.sidebar.date_input("Start date", key="global_start")
st.sidebar.date_input("End date", key="global_end")

available_targets = list_targets()
st.session_state.setdefault("target_universe", available_targets[:])
st.sidebar.multiselect(
    "Target universe",
    options=available_targets,
    default=st.session_state["target_universe"],
    key="target_universe",
    help="Targets included in the Today outlook strip and prediction tab dropdown.",
)

st.sidebar.subheader("Alert sensitivity")
sensitivity = st.sidebar.select_slider(
    "Conclusion thresholds",
    options=["Strict", "Default", "Sensitive"],
    value="Default",
    help=(
        "Strict: only fire on |z|≥2.5 with strong signals. "
        "Default: 2.0σ + r≥0.35. "
        "Sensitive: 1.5σ + r≥0.25 (expect more false positives)."
    ),
)
_SENSITIVITY = {
    "Strict":    ConclusionConfig(z_warning=2.5, z_critical=3.0,
                                  min_abs_correlation=0.45, max_granger_p=0.05),
    "Default":   ConclusionConfig(),
    "Sensitive": ConclusionConfig(z_warning=1.5, z_critical=2.0,
                                  min_abs_correlation=0.25, max_granger_p=0.15),
}
config = _SENSITIVITY[sensitivity]

if st.sidebar.button("🔄 Refresh data (clear cache)"):
    st.cache_data.clear()
    st.rerun()

start = pd.Timestamp(st.session_state["global_start"])
end = pd.Timestamp(st.session_state["global_end"])
if start >= end:
    st.sidebar.error("Start must be before end.")
    st.stop()


# ── Per-feature cadence ───────────────────────────────────────────────────────
#
# Maps feature name prefixes to (max_age_days, cadence_label).
# A feature is only "overdue" if age_days > max_age_days for its cadence.
# The rules are ordered most-specific first; first match wins.
_CADENCE_RULES: list[tuple[str, int, str]] = [
    ("FRED.GDP.",            120, "quarterly"),
    ("FRED.trade_balance.",   60, "monthly"),
    ("FRED.AUDUSD.",           5, "daily"),
    ("FRED.CNYUSD.",           5, "daily"),
    ("COMTRADE.",             90, "monthly"),
    ("port.USLAX.",           60, "monthly"),
    ("WCI.",                  12, "weekly"),
    ("BDI.",                   4, "daily"),
]
_DEFAULT_CADENCE = (4, "daily")


def _feature_cadence(feature_name: str) -> tuple[int, str]:
    """Return (max_expected_age_days, cadence_label) for a feature."""
    for prefix, max_age, label in _CADENCE_RULES:
        if feature_name.startswith(prefix):
            return max_age, label
    return _DEFAULT_CADENCE


def _annotate_freshness(feats: pd.DataFrame) -> pd.DataFrame:
    """Add expected_age, cadence, and overdue columns to a features frame."""
    today_ts = pd.Timestamp(date.today())
    feats = feats.copy()
    feats["age_days"] = (today_ts - pd.to_datetime(feats["date"])).dt.days
    cadence_info = feats["feature_name"].map(_feature_cadence)
    feats["expected_age"] = cadence_info.map(lambda t: t[0])
    feats["cadence"] = cadence_info.map(lambda t: t[1])
    feats["overdue"] = feats["age_days"] > feats["expected_age"]
    return feats


# ── Status banner (pulled from features freshness + recent alerts) ────────────
def render_status_banner() -> None:
    """One-line health summary at the top of every tab. Drives trust."""
    try:
        feats = load_features_latest()
    except Exception as exc:
        st.error(f"⛔ Database unreachable: {exc}")
        return

    if feats.empty:
        st.warning("⚠️ No features in database — has the normalizer run?")
        return

    feats = _annotate_freshness(feats)
    overdue_count = int(feats["overdue"].sum())
    total = len(feats)

    if overdue_count == 0:
        st.success(
            f"🟢 All systems green — {total} features current "
            f"(latest {feats['date'].max()})"
        )
    elif overdue_count <= 3:
        names = ", ".join(feats.loc[feats["overdue"], "feature_name"].tolist())
        st.warning(
            f"🟡 {overdue_count}/{total} features overdue: {names}. "
            f"See **Health** tab."
        )
    else:
        st.error(
            f"🔴 {overdue_count}/{total} features overdue. "
            f"Pipeline likely broken. See **Health** tab."
        )


# ── Conclusion rendering ──────────────────────────────────────────────────────
_DOT = {"critical": "🔴", "warning": "🟡", "info": "⚪"}


def render_conclusion_card(conclusion: Conclusion, idx: int) -> None:
    """Sentence + 'Show evidence' button. Click hops to Signals/Predictions
    pre-filtered to this conclusion's pair."""
    dot = _DOT.get(conclusion.severity, "⚪")
    cols = st.columns([20, 3])
    with cols[0]:
        st.markdown(f"**{dot} {conclusion.headline}**")
        nums = conclusion.supporting_numbers
        # Show one line of supporting numbers as a caption — terse, not a
        # second sentence. The full dict is in the link target.
        if conclusion.conclusion_type == "threshold_breach":
            st.caption(
                f"r={nums.get('pearson_r', 0):+.2f}  "
                f"lag={nums.get('lag_days', 0)}d  "
                f"window={nums.get('window_days', 0)}d  "
                f"n={nums.get('sample_size', 0)}"
            )
        elif conclusion.conclusion_type == "regime_change":
            st.caption(
                f"prior r={nums.get('prior_correlation', 0):+.2f}  →  "
                f"recent r={nums.get('recent_correlation', 0):+.2f}  "
                f"(Δ={nums.get('delta', 0):+.2f})"
            )
        elif conclusion.conclusion_type == "model_extreme":
            pz = nums.get("predicted_z")
            pz_str = f"{pz:+.2f}σ" if pz is not None else "n/a"
            st.caption(
                f"horizon={nums.get('horizon_days', 0)}d  "
                f"predicted_z={pz_str}  "
                f"model={nums.get('model_version', '?')}"
            )
    with cols[1]:
        if conclusion.evidence_link:
            st.button(
                "Show evidence",
                key=f"evidence_{idx}",
                on_click=_navigate,
                args=(conclusion.evidence_link,),
                use_container_width=True,
            )


# ── Tab 1: Today ──────────────────────────────────────────────────────────────
def render_today_tab() -> None:
    st.header("Today")
    st.caption(
        "If you only have 30 seconds, this is the page. Conclusions come from "
        "the signals + predictions tables; cards are at-a-glance target outlook; "
        "the delta table catches anything that moved meaningfully overnight."
    )

    # ── Section A: Headline conclusions ─────────────────────────────────────
    st.subheader("A. Headline conclusions")

    feats_latest = load_features_latest()
    signals_latest = load_signals_latest()
    signals_hist = load_signals_history()
    preds_latest = load_predictions_latest()
    preds_hist = load_predictions_history()

    conclusions = generate_conclusions(
        features_today=feats_latest,
        signals=signals_latest,
        signals_history=signals_hist,
        predictions_today=preds_latest,
        predictions_history=preds_hist,
        config=config,
    )

    if conclusions and conclusions[0].conclusion_type == "no_signal":
        st.info(conclusions[0].headline)
    else:
        for i, c in enumerate(conclusions):
            render_conclusion_card(c, i)
            st.divider()

    # ── Section B: Target outlook strip ─────────────────────────────────────
    st.subheader("B. Target outlook")
    targets = st.session_state["target_universe"]
    if not targets:
        st.info("No targets selected in sidebar.")
    else:
        cols = st.columns(min(len(targets), 4))
        for i, t in enumerate(targets):
            with cols[i % len(cols)]:
                _render_target_card(t, preds_latest, preds_hist)

    # ── Section C: What changed since yesterday ─────────────────────────────
    st.subheader("C. What changed since yesterday")
    changes = _compute_daily_changes()
    if changes.empty:
        st.info("Not enough recent data to compute day-over-day changes.")
    else:
        st.dataframe(changes.head(5), use_container_width=True, hide_index=True)


def _render_target_card(
    target: str,
    preds_latest: pd.DataFrame,
    preds_hist: pd.DataFrame,
) -> None:
    """One card per target: current value, model prediction, percentile,
    60-day sparkline. Hand-rolled in markdown — st.metric is too rigid for
    the 'value + prediction + percentile' triple."""
    target_hist = load_target_history(target)
    pred_row = preds_latest[preds_latest["target_name"] == target]

    if target_hist.empty:
        st.markdown(f"**{target}**\n\n_no target data_")
        return

    current = float(target_hist["value"].iloc[-1])
    last_date = target_hist["date"].iloc[-1].date()

    if pred_row.empty:
        pred_str = "n/a"
        pct_str = ""
        pred_value = None
        horizon = None
    else:
        # If multiple horizons exist, show the longest (most "interesting").
        pr = pred_row.sort_values("horizon_days").iloc[-1]
        pred_value = float(pr["predicted_value"])
        horizon = int(pr["horizon_days"])
        hist = preds_hist[
            (preds_hist["target_name"] == target)
            & (preds_hist["horizon_days"] == horizon)
        ]["predicted_value"].dropna()
        if len(hist) >= 30:
            pct = float((hist < pred_value).mean()) * 100
            pct_str = f"({_ordinal(int(pct))} pct)"
        else:
            pct_str = "(thin history)"
        pred_str = f"{pred_value:+.4f} over {horizon}d"

    st.markdown(f"### {target}")
    st.markdown(
        f"**Current:** `{current:+.4f}`  \n"
        f"**Model:** `{pred_str}` {pct_str}  \n"
        f"_as of {last_date}_"
    )

    # 60-day sparkline of the target with the predicted point extended.
    spark = target_hist.tail(60).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spark["date"], y=spark["value"],
        mode="lines", line=dict(width=1.5),
        name=target, showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
    ))
    if pred_value is not None and horizon is not None:
        future_x = spark["date"].iloc[-1] + pd.Timedelta(days=horizon)
        fig.add_trace(go.Scatter(
            x=[spark["date"].iloc[-1], future_x],
            y=[current, pred_value],
            mode="lines+markers",
            line=dict(dash="dot", width=1.5),
            name="prediction", showlegend=False,
            hovertemplate="prediction<br>%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        height=110,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})

    if st.button("Expand →", key=f"expand_{target}", use_container_width=True):
        _navigate({"tab": "Predictions", "filters": {"target_name": target}})
        st.rerun()


_COUNT_UNITS: list[tuple[str, str]] = [
    ("vessels", "vessels"),
    ("teu", "TEU"),
    ("calls", "calls"),
    ("count", ""),
]


def _fmt_change(feature_name: str, today_val: float, prev_val: float) -> str:
    """Format a raw-value change as either % (prices/indices) or abs+unit (counts)."""
    if pd.isna(today_val) or pd.isna(prev_val):
        return "n/a"
    fname = feature_name.lower()
    for pattern, unit in _COUNT_UNITS:
        if pattern in fname:
            delta = int(round(today_val - prev_val))
            return f"{delta:+d} {unit}".strip()
    # Values already in rate/normalized form (|val| < 1) — e.g. momentum, returns,
    # log-ratios. % change on a near-zero base produces nonsense like 3455%.
    if abs(prev_val) < 1.0:
        return f"{today_val - prev_val:+.4f}"
    pct = (today_val - prev_val) / abs(prev_val) * 100
    return f"{pct:+.1f}%"


def _compute_daily_changes() -> pd.DataFrame:
    """Current value + 1-day and 7-day changes per feature, sorted by |1-day change|."""
    df = load_features_for_delta(days_back=10)
    if df.empty:
        return pd.DataFrame()

    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return pd.DataFrame()

    today_d = dates[-1]
    prev_1d = dates[-2]
    week_ago_target = today_d - pd.Timedelta(days=7)
    prev_7d_candidates = [d for d in dates if d <= week_ago_target]
    prev_7d = max(prev_7d_candidates) if prev_7d_candidates else None

    def _vals(d) -> pd.Series:
        return df[df["date"] == d].set_index("feature_name")["value"]

    today_vals = _vals(today_d)
    prev_1d_vals = _vals(prev_1d)
    prev_7d_vals = _vals(prev_7d) if prev_7d else pd.Series(dtype=float)

    common = today_vals.index.intersection(prev_1d_vals.index)
    rows = []
    for feat in common:
        tv = float(today_vals[feat])
        p1 = float(prev_1d_vals[feat])
        p7 = float(prev_7d_vals[feat]) if feat in prev_7d_vals.index else float("nan")
        # Sort key: absolute % change (or abs delta for counts) for 1-day move
        fname = feat.lower()
        is_count = any(p in fname for p, _ in _COUNT_UNITS)
        sort_key = abs(tv - p1) if is_count or p1 == 0 else abs((tv - p1) / abs(p1))
        rows.append({
            "Feature": feat,
            "Current value": f"{tv:,.4g}",
            "1-day change": _fmt_change(feat, tv, p1),
            "7-day change": _fmt_change(feat, tv, p7),
            "_sort": sort_key,
        })

    out = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
    return out.reset_index(drop=True)


# ── Tab 2: Signals ────────────────────────────────────────────────────────────
def render_signals_tab() -> None:
    st.header("Signals")
    st.caption(
        "The lead-lag substrate. Filter the grid, click a row to see its "
        "evidence (overlay plot + scatter + stability dots). Don't trust a "
        "single average correlation — watch the three sub-window dots."
    )

    sig = load_signals_latest()
    sig_hist = load_signals_history()

    if sig.empty:
        st.info("No signals yet. Run `python -m signals.builder` first.")
        return

    # ── Filters ─────────────────────────────────────────────────────────────
    nav_filters = st.session_state.get("nav_filters", {}) or {}
    min_r_default = float(_ss("signals_min_r", 0.30))
    max_p_default = float(_ss("signals_max_p", 0.10))
    regime_default = bool(_ss("signals_regime_only", False))

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        min_r = st.slider("Min |pearson r|", 0.0, 1.0, min_r_default, 0.05,
                          key="signals_min_r")
    with c2:
        max_p = st.slider("Max granger p", 0.0, 1.0, max_p_default, 0.01,
                          key="signals_max_p")
    with c3:
        regime_only = st.checkbox(
            "Regime change only (|Δr| > 0.20 vs. prior)",
            value=regime_default, key="signals_regime_only",
        )

    grid = sig.copy()
    grid = grid[grid["pearson_r"].abs() >= min_r]
    grid = grid[grid["granger_p"].isna() | (grid["granger_p"] <= max_p)]

    if regime_only and not sig_hist.empty:
        # For each (feature, target, window), compare the latest 30d mean to
        # the prior 90d mean; keep where the absolute shift exceeds 0.20.
        shifted = _regime_shift_table(sig_hist, lookback_days=30,
                                      threshold=0.20)
        if not shifted.empty:
            grid = grid.merge(
                shifted[["feature_name", "target_name", "window_days"]],
                on=["feature_name", "target_name", "window_days"],
                how="inner",
            )
        else:
            grid = grid.iloc[0:0]

    # Optional pre-filter from a Today conclusion link.
    if nav_filters.get("feature_name"):
        grid = grid[grid["feature_name"] == nav_filters["feature_name"]]
    if nav_filters.get("target_name"):
        grid = grid[grid["target_name"] == nav_filters["target_name"]]

    st.caption(
        f"{len(grid)} of {len(sig)} signals match. "
        f"Click a row in the table below to render its evidence."
    )

    if grid.empty:
        st.info("No signals match the current filters.")
        return

    grid_display = grid.copy()
    grid_display = grid_display.sort_values(
        "pearson_r", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)

    selected_rows = st.dataframe(
        grid_display.style.format({
            "pearson_r":  "{:+.3f}",
            "spearman_r": "{:+.3f}",
            "granger_p":  "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    sel = (selected_rows.get("selection", {}) if hasattr(selected_rows, "get")
           else getattr(selected_rows, "selection", {}))
    selected_idx = (sel or {}).get("rows", [])
    if not selected_idx:
        # Auto-select the top row when arriving from a Today link, so the
        # evidence panel renders immediately. The user can still click another
        # row to switch.
        if nav_filters.get("feature_name") and not grid_display.empty:
            chosen = grid_display.iloc[0]
        else:
            st.info("Select a row to see evidence plots.")
            return
    else:
        chosen = grid_display.iloc[selected_idx[0]]

    _render_signal_evidence(chosen, sig_hist)


def _regime_shift_table(
    sig_hist: pd.DataFrame, lookback_days: int, threshold: float
) -> pd.DataFrame:
    """For each (feature, target, window), |recent r mean - prior r mean| > thr."""
    if sig_hist.empty:
        return pd.DataFrame()
    today_d = pd.Timestamp(sig_hist["as_of_date"].max())
    cutoff = today_d - pd.Timedelta(days=lookback_days)

    df = sig_hist.copy()
    df["as_of_ts"] = pd.to_datetime(df["as_of_date"])
    grouped = df.groupby(["feature_name", "target_name", "window_days"])

    rows = []
    for keys, g in grouped:
        recent = g[g["as_of_ts"] >= cutoff]["pearson_r"]
        prior = g[g["as_of_ts"] < cutoff]["pearson_r"]
        if len(recent) < 5 or len(prior) < 30:
            continue
        if abs(recent.mean() - prior.mean()) >= threshold:
            rows.append({
                "feature_name": keys[0],
                "target_name":  keys[1],
                "window_days":  keys[2],
                "delta":        float(recent.mean() - prior.mean()),
            })
    return pd.DataFrame(rows)


def _render_signal_evidence(chosen: pd.Series, sig_hist: pd.DataFrame) -> None:
    feature = chosen["feature_name"]
    target = chosen["target_name"]
    lag = int(chosen["lag_days"])

    st.markdown(
        f"### Evidence: `{feature}` (lag {lag:+d}d) ↔ `{target}`"
    )
    granger_p_val = chosen["granger_p"]
    granger_str = "n/a" if pd.isna(granger_p_val) else f"{float(granger_p_val):.3f}"
    st.caption(
        f"r={float(chosen['pearson_r']):+.3f}  "
        f"granger_p={granger_str}  "
        f"n={int(chosen['sample_size'])}  window={int(chosen['window_days'])}d"
    )

    # ── Stability indicator: r computed over 3 non-overlapping sub-windows.
    stability = signal_stability(sig_hist, feature, target, n_windows=3)
    if stability:
        st.markdown("**Stability across history (3 sub-windows):**")
        fig_stab = go.Figure()
        fig_stab.add_trace(go.Scatter(
            x=stability,
            y=[0] * len(stability),
            mode="markers",
            marker=dict(size=20, color=["#2563eb", "#9333ea", "#16a34a"][:len(stability)]),
            text=[f"{r:+.2f}" for r in stability],
            textposition="top center",
            showlegend=False,
            hovertemplate="r=%{x:+.3f}<extra></extra>",
        ))
        fig_stab.add_vline(x=0, line_dash="dot", line_color="grey")
        fig_stab.update_layout(
            height=120,
            xaxis=dict(range=[-1, 1], title="pearson r"),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_stab, use_container_width=True,
                        config={"displayModeBar": False})
        spread = max(stability) - min(stability)
        if spread < 0.15:
            st.success(f"🟢 Stable: range {spread:.2f} across sub-windows.")
        elif spread < 0.35:
            st.warning(f"🟡 Mixed: range {spread:.2f}.")
        else:
            st.error(
                f"🔴 Unstable: range {spread:.2f}. "
                f"The headline r is averaging over distinct regimes."
            )
    else:
        st.caption("Not enough history yet for a stability check.")

    # ── Time-series overlay (lag-shifted) ──────────────────────────────────
    feat_hist = load_features_history(start.date(), end.date())
    target_hist = load_target_history(target)
    if feat_hist.empty or target_hist.empty:
        st.info("Not enough overlap to plot.")
        return

    f = feat_hist[feat_hist["feature_name"] == feature][["date", "z_score"]]
    f = f.set_index("date")["z_score"].dropna()
    f_shifted = f.shift(lag)

    t_series = target_hist.set_index("date")["value"].dropna()
    t_z = (t_series - t_series.mean()) / t_series.std(ddof=0)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=f_shifted.index, y=f_shifted.values,
                                name=f"{feature} (lag {lag:+d}d, z)",
                                mode="lines"))
    fig_ts.add_trace(go.Scatter(x=t_z.index, y=t_z.values,
                                name=f"{target} (z-normalised)",
                                mode="lines"))
    fig_ts.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=10, b=10),
        legend=dict(orientation="h", y=-0.2),
        yaxis_title="z-score",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Scatter + regression line ──────────────────────────────────────────
    aligned = pd.concat(
        [f_shifted.rename("x"), t_z.rename("y")], axis=1, join="inner"
    ).dropna()
    if len(aligned) >= 10:
        fig_sc = px.scatter(
            aligned, x="x", y="y", trendline="ols",
            labels={"x": f"{feature} (lag {lag:+d}d, z)", "y": f"{target} (z)"},
        )
        fig_sc.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)


# ── Tab 3: Predictions ────────────────────────────────────────────────────────
def render_predictions_tab() -> None:
    st.header("Predictions")
    st.caption(
        "One target per view. Top: actuals + walk-forward predictions. "
        "Middle: residual MAE band — your 'is the model still working' check. "
        "Bottom: today's per-feature contribution proxy."
    )

    targets = st.session_state["target_universe"]
    if not targets:
        st.info("No targets selected in the sidebar.")
        return

    nav_target = (st.session_state.get("nav_filters") or {}).get("target_name")
    default_idx = (
        targets.index(nav_target)
        if nav_target in targets else 0
    )
    target = st.selectbox("Target", options=targets, index=default_idx)

    preds = load_predictions_history(target)
    if preds.empty:
        st.warning(
            f"No predictions for `{target}`. Run `python -m models.trainer`."
        )
        return

    horizons = sorted(preds["horizon_days"].unique())
    horizon = st.selectbox("Horizon (days)", options=horizons, index=0)
    pred_slice = preds[preds["horizon_days"] == horizon].sort_values("date")

    # ── Health badge: residual MAE last 30d vs. trailing 180d ──────────────
    actuals = load_target_history(target)
    actuals = actuals[actuals["horizon_days"] == horizon]
    merged = pred_slice.merge(
        actuals[["date", "value"]].rename(columns={"value": "actual"}),
        on="date", how="left",
    )
    merged["residual"] = merged["actual"] - merged["predicted_value"]

    realized = merged.dropna(subset=["residual"])
    if len(realized) >= 30:
        recent_mae = realized.tail(30)["residual"].abs().mean()
        long_mae = realized.tail(180)["residual"].abs().mean() \
            if len(realized) >= 180 else realized["residual"].abs().mean()
        ratio = recent_mae / long_mae if long_mae > 0 else float("nan")
        if ratio < 1.15:
            st.success(
                f"🟢 Model healthy — recent 30d MAE {recent_mae:.4f} "
                f"vs trailing MAE {long_mae:.4f} (ratio {ratio:.2f})."
            )
        elif ratio < 1.5:
            st.warning(
                f"🟡 Model degrading — recent MAE {recent_mae:.4f} "
                f"is {ratio:.2f}× the trailing baseline. "
                f"Today's prediction may be less reliable."
            )
        else:
            st.error(
                f"🔴 Model unreliable — recent MAE {recent_mae:.4f} "
                f"is {ratio:.2f}× trailing. Treat predictions with caution."
            )
    else:
        st.info(
            f"Only {len(realized)} realized residuals so far — health check "
            f"needs ≥30 to be meaningful."
        )

    # ── Plot 1: actuals + predictions overlay (with predicted_z fan as a
    # crude band proxy when no quantile preds are stored) ──────────────────
    fig_pred = go.Figure()
    if not actuals.empty:
        fig_pred.add_trace(go.Scatter(
            x=actuals["date"], y=actuals["value"],
            mode="lines", name="actual",
            line=dict(width=2),
        ))
    fig_pred.add_trace(go.Scatter(
        x=pred_slice["date"], y=pred_slice["predicted_value"],
        mode="lines", name="prediction (walk-forward)",
        line=dict(dash="dot"),
    ))
    fig_pred.update_layout(
        height=320, margin=dict(l=20, r=20, t=10, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"{target} (horizon {horizon}d) — actual vs. walk-forward prediction",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Plot 2: residuals + rolling MAE band ────────────────────────────────
    if not realized.empty:
        roll = realized.set_index("date")["residual"]
        roll_mae = roll.abs().rolling(30, min_periods=10).mean()
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=roll.index, y=roll.values, mode="markers",
            name="residual", marker=dict(size=4, opacity=0.6),
        ))
        fig_res.add_trace(go.Scatter(
            x=roll_mae.index, y=roll_mae.values, mode="lines",
            name="30d rolling MAE", line=dict(width=2),
        ))
        fig_res.add_trace(go.Scatter(
            x=roll_mae.index, y=-roll_mae.values, mode="lines",
            name="-30d rolling MAE",
            line=dict(width=2), showlegend=False,
        ))
        fig_res.add_hline(y=0, line_dash="dot", line_color="grey")
        fig_res.update_layout(
            height=260, margin=dict(l=20, r=20, t=30, b=10),
            title="Residuals over time (MAE band = ±rolling 30d MAE)",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("No realized residuals yet to render.")

    # ── Plot 3: today's per-feature contribution proxy ──────────────────────
    #
    # The trainer doesn't currently persist coefficients to the DB, so we
    # can't reconstruct per-feature SHAP/contribution server-side. As a
    # principled proxy: show the strongest signals for THIS target — these
    # are the features the model is most likely keying on. Document the
    # caveat plainly so users don't read it as a true contribution chart.
    st.markdown("**Today's likely drivers** (proxy from `signals` table)")
    st.caption(
        "We don't persist trained coefficients, so this shows the strongest "
        "lead-lag relationships for this target × today's feature z, "
        "not true model contributions. Replace with SHAP or "
        "coef × current-z when the trainer persists models."
    )

    sig_latest = load_signals_latest()
    feats_latest = load_features_latest()
    drivers = sig_latest[sig_latest["target_name"] == target].copy()
    if drivers.empty:
        st.info("No signals for this target.")
    else:
        drivers = drivers.merge(
            feats_latest[["feature_name", "z_score"]].rename(
                columns={"z_score": "current_z"}
            ),
            on="feature_name", how="left",
        )
        drivers["contribution_proxy"] = (
            drivers["pearson_r"] * drivers["current_z"]
        )
        drivers = drivers.dropna(subset=["contribution_proxy"])
        drivers = drivers.reindex(
            drivers["contribution_proxy"].abs()
            .sort_values(ascending=False).index
        ).head(10)
        fig_imp = px.bar(
            drivers, x="contribution_proxy", y="feature_name",
            orientation="h", color="contribution_proxy",
            color_continuous_scale="RdBu", color_continuous_midpoint=0,
            labels={"contribution_proxy": "r × current z",
                    "feature_name": ""},
            hover_data=["pearson_r", "current_z", "lag_days"],
        )
        fig_imp.update_layout(
            height=380, margin=dict(l=20, r=20, t=10, b=10),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ── Tab 4: Explore (legacy correlation tool) ──────────────────────────────────
def render_explore_tab() -> None:
    st.header("Explore")
    st.caption(
        "Ad-hoc correlation / lead-lag / Granger explorer. "
        "Use when Today + Signals don't quite answer the question."
    )

    all_features = list_features()
    if not all_features:
        st.info("No features in database.")
        return

    default_features = [
        f for f in (
            "BDI.daily_close",
            "FRED.INDPRO",
            "port.CNSHA.vessels_in_port",
            "air.cargo_flights.daily",
        ) if f in all_features
    ] or all_features[: min(5, len(all_features))]

    selected = st.multiselect(
        "Features", options=all_features, default=default_features,
        key="explore_features",
    )
    if len(selected) < 2:
        st.info("Pick at least two features.")
        return

    value_col = st.radio(
        "Value column", options=("value", "z_score"), index=1, horizontal=True,
        key="explore_valcol",
    )

    df = load_features_history(start.date(), end.date())
    df = df[df["feature_name"].isin(selected)]
    wide = df.pivot(index="date", columns="feature_name", values=value_col)
    wide = wide.sort_index()

    st.markdown("**Lead/lag shifts**")
    cols = st.columns(min(len(selected), 4))
    shifts: dict[str, int] = {}
    for i, feat in enumerate(selected):
        with cols[i % len(cols)]:
            shifts[feat] = st.slider(
                feat, -30, 30, 0, key=f"explore_shift_{feat}"
            )

    shifted = wide.copy()
    for col, n in shifts.items():
        if n and col in shifted.columns:
            shifted[col] = shifted[col].shift(n)
    shifted = shifted.dropna(how="all")

    st.subheader("Series preview")
    fig_ts = px.line(
        shifted.reset_index().melt(
            id_vars="date", var_name="feature", value_name="v"
        ),
        x="date", y="v", color="feature",
        labels={"v": value_col},
    )
    fig_ts.update_layout(height=320, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Correlation matrix")
    corr = shifted.corr(method="pearson", min_periods=20)
    fig_hm = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig_hm.update_layout(height=480, margin=dict(l=80, r=40, t=10, b=80))
    st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Lag sweep")
    target_feat = st.selectbox(
        "Target feature", options=selected, index=0, key="explore_target"
    )
    max_lag = st.slider("Max lag (± days)", 5, 60, 21, 1, key="explore_maxlag")
    sweep = _lag_sweep(wide, target_feat, max_lag)
    if not sweep.empty:
        fig_sweep = px.line(
            sweep, x="lag", y="r", color="feature",
            labels={"lag": "shift of feature (days)", "r": "Pearson r"},
        )
        fig_sweep.add_hline(y=0, line_dash="dot", line_color="grey")
        fig_sweep.update_layout(height=300)
        st.plotly_chart(fig_sweep, use_container_width=True)

    if _HAS_GRANGER:
        st.subheader("Granger causality")
        c1, c2, c3 = st.columns([3, 3, 2])
        with c1:
            cause = st.selectbox(
                "Cause", options=selected, index=0, key="explore_cause"
            )
        with c2:
            eff_opts = [f for f in selected if f != cause] or selected
            effect = st.selectbox(
                "Effect", options=eff_opts, index=0, key="explore_effect"
            )
        with c3:
            mlag = st.number_input("Max lag", 1, 30, 7, 1, key="explore_glag")

        pair = wide[[effect, cause]].dropna()
        if len(pair) < 30:
            st.warning(f"Only {len(pair)} overlapping rows — need ≥30.")
        elif cause == effect:
            st.warning("Pick two distinct features.")
        else:
            try:
                with st.spinner("Running Granger test…"):
                    res = grangercausalitytests(
                        pair.values, maxlag=int(mlag), verbose=False
                    )
                gc = pd.DataFrame([
                    {"lag": lag, "F": s["ssr_ftest"][0],
                     "p_value": s["ssr_ftest"][1]}
                    for lag, (s, _) in res.items()
                ])
                fig_gc = px.bar(
                    gc, x="lag", y="p_value",
                    labels={"p_value": "p (ssr F-test)"},
                )
                fig_gc.add_hline(
                    y=0.05, line_dash="dash", line_color="red",
                    annotation_text="α=0.05",
                    annotation_position="top right",
                )
                fig_gc.update_layout(height=260)
                st.plotly_chart(fig_gc, use_container_width=True)
            except Exception as exc:
                st.error(f"Granger test failed: {exc}")
    else:
        st.info("statsmodels not installed — Granger test disabled.")


def _lag_sweep(wide: pd.DataFrame, target: str, max_lag: int) -> pd.DataFrame:
    rows = []
    for feat in wide.columns:
        if feat == target:
            continue
        for lag in range(-max_lag, max_lag + 1):
            shifted = wide[feat].shift(lag)
            df = pd.concat([wide[target], shifted], axis=1).dropna()
            if len(df) < 20:
                continue
            r = df.iloc[:, 0].corr(df.iloc[:, 1])
            rows.append({"feature": feat, "lag": lag, "r": r})
    return pd.DataFrame(rows)


# ── Tab 5: Health ─────────────────────────────────────────────────────────────
def render_health_tab() -> None:
    st.header("Health")
    st.caption(
        "Pipeline freshness and alert flow. The status banner at the top of "
        "every tab is computed from the same data — this view is the detail."
    )

    feats = load_features_latest()
    if feats.empty:
        st.error("No features in database.")
        return

    feats = _annotate_freshness(feats).sort_values("age_days", ascending=False)

    st.subheader("Feature freshness")
    cnt_current = int((~feats["overdue"]).sum())
    cnt_overdue = int(feats["overdue"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total features", len(feats))
    c2.metric("Current", cnt_current)
    c3.metric("Overdue", cnt_overdue,
              delta=None if cnt_overdue == 0 else "needs attention")

    display = feats[["feature_name", "cadence", "date", "age_days",
                      "expected_age", "overdue", "z_score"]].reset_index(drop=True)

    def _age_color(row):
        color = "color: #cc3300" if row["overdue"] else ""
        return [color] * len(row)

    st.dataframe(
        display.style
        .apply(_age_color, axis=1)
        .format({"z_score": "{:+.2f}"}),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Recent alerts (last 14 days)")
    alerts = load_alerts_recent(days=14)
    if alerts.empty:
        st.info("No alerts in the last 14 days.")
    else:
        st.dataframe(
            alerts[[
                "triggered_at", "as_of_date", "severity", "alert_type",
                "subject", "feature_name", "target_name", "z_score",
                "pearson_r",
            ]],
            use_container_width=True, hide_index=True,
        )

    st.subheader("Model retraining audit")
    preds = load_predictions_latest()
    if preds.empty:
        st.info("No model predictions yet.")
    else:
        retrain = preds.copy()
        retrain["age_days"] = (
            pd.Timestamp(date.today()) - pd.to_datetime(retrain["date"])
        ).dt.days
        st.dataframe(
            retrain[[
                "target_name", "horizon_days", "model_version",
                "date", "age_days", "predicted_value", "predicted_z",
            ]].sort_values(["target_name", "horizon_days"]).reset_index(drop=True)
            .style.format({"predicted_value": "{:+.4f}",
                           "predicted_z": "{:+.2f}"}),
            use_container_width=True, hide_index=True,
        )


# ── Tab routing ───────────────────────────────────────────────────────────────
TABS = ["Today", "Signals", "Predictions", "Explore", "Health"]

st.title("Trade Analysis — Daily Monitor")
render_status_banner()

# Honour cross-tab navigation requests, but only re-route once. After the
# user manually clicks a different tab, we don't want to keep snapping them
# back to the navigated one.
nav_target = st.session_state.pop("nav_to_tab", None)
if nav_target and nav_target in TABS:
    # Streamlit doesn't expose programmatic tab switching; use radio + filter
    # state instead. Setting nav_filters before the radio renders means the
    # destination tab will pick them up. We render a notice so the user
    # understands why the radio jumped.
    st.session_state["_active_tab"] = nav_target
    st.toast(f"Jumped to {nav_target} tab — filters pre-applied.")

# Persist active tab so navigation actions survive reruns.
_ss("_active_tab", "Today")
active_tab = st.radio(
    "View", options=TABS,
    index=TABS.index(st.session_state["_active_tab"]),
    horizontal=True, label_visibility="collapsed", key="_active_tab",
)

st.divider()

if active_tab == "Today":
    render_today_tab()
elif active_tab == "Signals":
    render_signals_tab()
elif active_tab == "Predictions":
    render_predictions_tab()
elif active_tab == "Explore":
    render_explore_tab()
elif active_tab == "Health":
    render_health_tab()
