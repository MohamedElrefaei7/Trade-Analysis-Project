"""
correlation.py — Streamlit correlation & lead/lag explorer.

Reads the `features` table, pivots it into a date × feature matrix, and lets the
user:
  • pick a subset of features
  • shift selected features by ±N days to test lead/lag relationships
  • view a Pearson correlation heatmap
  • view a pairwise lag sweep (best correlation across lags) for a target feature
  • run a Granger causality test between two selected features

Run:
    streamlit run dashboard/correlation.py
"""

import os
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_GRANGER = True
except Exception:
    _HAS_GRANGER = False


# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://admin:password@localhost:5432/mydb"
)

st.set_page_config(
    page_title="Trade Analysis — Correlation Explorer",
    layout="wide",
)


# ── Data access ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=300)
def list_features() -> list[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT feature_name FROM features ORDER BY feature_name")
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=300)
def load_features(
    feature_names: tuple[str, ...],
    value_col: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return a wide DataFrame indexed by date with one column per feature."""
    if not feature_names:
        return pd.DataFrame()

    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(
                f"""
                SELECT date, feature_name, {value_col} AS v
                FROM features
                WHERE feature_name = ANY(:names)
                  AND date BETWEEN :start AND :end
                  AND {value_col} IS NOT NULL
                ORDER BY date
                """
            ),
            conn,
            params={
                "names": list(feature_names),
                "start": start_date.date(),
                "end": end_date.date(),
            },
        )

    if df.empty:
        return df

    wide = df.pivot(index="date", columns="feature_name", values="v")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def apply_shifts(df: pd.DataFrame, shifts: dict[str, int]) -> pd.DataFrame:
    """Shift each column by `shifts[col]` days. Positive shift = future → past
    (i.e. that feature leads the others by N days)."""
    out = df.copy()
    for col, n in shifts.items():
        if n and col in out.columns:
            out[col] = out[col].shift(n)
    return out


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Data")

all_features = list_features()
if not all_features:
    st.error("No features found. Has the normalizer run against the database?")
    st.stop()

default_features = [
    f
    for f in (
        "BDI.daily_close",
        "FRED.INDPRO",
        "port.CNSHA.vessels_in_port",
        "air.cargo_flights.daily",
    )
    if f in all_features
] or all_features[: min(5, len(all_features))]

selected = st.sidebar.multiselect(
    "Features",
    options=all_features,
    default=default_features,
    help="Pick 2+ features to compare.",
)

value_col = st.sidebar.radio(
    "Value column",
    options=("value", "z_score"),
    index=1,
    help="z_score is standardised; value is raw.",
)

today = pd.Timestamp.utcnow().normalize()
start = st.sidebar.date_input("Start date", value=(today - pd.Timedelta(days=365)).date())
end = st.sidebar.date_input("End date", value=today.date())

if pd.Timestamp(start) >= pd.Timestamp(end):
    st.sidebar.error("Start must be before end.")
    st.stop()

st.sidebar.header("Lead / Lag")
st.sidebar.caption(
    "Positive N shifts a feature forward in time (i.e. treats it as "
    "leading the others by N days)."
)

shifts: dict[str, int] = {}
if selected:
    for feat in selected:
        shifts[feat] = st.sidebar.slider(
            feat,
            min_value=-30,
            max_value=30,
            value=0,
            step=1,
            key=f"shift_{feat}",
        )


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Trade Analysis — Correlation Explorer")
st.caption(
    "Pairwise Pearson correlations and Granger causality between normalised "
    "features, with optional lead/lag shifts."
)

if len(selected) < 2:
    st.info("Pick at least two features in the sidebar.")
    st.stop()

wide = load_features(
    tuple(selected), value_col, pd.Timestamp(start), pd.Timestamp(end)
)

if wide.empty:
    st.warning("No rows in the selected range.")
    st.stop()

wide = apply_shifts(wide, shifts)
wide = wide.dropna(how="all")

st.subheader("Series preview")
fig_ts = px.line(
    wide.reset_index().melt(id_vars="date", var_name="feature", value_name="v"),
    x="date",
    y="v",
    color="feature",
    labels={"v": value_col},
)
fig_ts.update_layout(height=360, legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig_ts, use_container_width=True)

# ── Correlation heatmap ──
st.subheader("Correlation matrix")
corr = wide.corr(method="pearson", min_periods=20)
n_obs = wide.dropna().shape[0]
st.caption(f"Pearson correlation — {n_obs} overlapping observations after shifts.")

fig_hm = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        reversescale=True,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    )
)
fig_hm.update_layout(height=520, margin=dict(l=80, r=40, t=30, b=80))
st.plotly_chart(fig_hm, use_container_width=True)

# ── Lag sweep ──
st.subheader("Lag sweep")
st.caption(
    "For a chosen target, find the shift (in days) of each other feature that "
    "maximises |correlation| with the target."
)

target = st.selectbox("Target feature", options=selected, index=0)
max_lag = st.slider("Max lag (± days)", min_value=5, max_value=60, value=21, step=1)

# Reload unshifted data for an honest sweep so sidebar shifts don't compound.
unshifted = load_features(
    tuple(selected), value_col, pd.Timestamp(start), pd.Timestamp(end)
)

def _sweep(
    target_series: pd.Series, other: pd.Series, lags: Iterable[int]
) -> pd.DataFrame:
    rows = []
    for lag in lags:
        shifted = other.shift(lag)
        df = pd.concat([target_series, shifted], axis=1).dropna()
        if len(df) < 20:
            rows.append({"lag": lag, "r": np.nan, "n": len(df)})
            continue
        r = df.iloc[:, 0].corr(df.iloc[:, 1])
        rows.append({"lag": lag, "r": r, "n": len(df)})
    return pd.DataFrame(rows)


lag_range = range(-max_lag, max_lag + 1)
sweep_frames = []
summary_rows = []
for feat in selected:
    if feat == target:
        continue
    s = _sweep(unshifted[target], unshifted[feat], lag_range)
    s["feature"] = feat
    sweep_frames.append(s)
    if s["r"].notna().any():
        best_idx = s["r"].abs().idxmax()
        summary_rows.append(
            {
                "feature": feat,
                "best_lag": int(s.loc[best_idx, "lag"]),
                "best_r": float(s.loc[best_idx, "r"]),
                "n": int(s.loc[best_idx, "n"]),
            }
        )

if sweep_frames:
    sweep_df = pd.concat(sweep_frames, ignore_index=True)
    fig_sweep = px.line(
        sweep_df, x="lag", y="r", color="feature",
        labels={"lag": "shift of feature (days)", "r": "Pearson r"},
    )
    fig_sweep.add_hline(y=0, line_dash="dot", line_color="grey")
    fig_sweep.update_layout(height=360)
    st.plotly_chart(fig_sweep, use_container_width=True)

    st.dataframe(
        pd.DataFrame(summary_rows).sort_values("best_r", key=lambda s: s.abs(), ascending=False),
        use_container_width=True,
    )

# ── Granger causality ──
st.subheader("Granger causality")
if not _HAS_GRANGER:
    st.info("statsmodels not installed — Granger test disabled.")
else:
    c1, c2, c3 = st.columns([3, 3, 2])
    with c1:
        cause = st.selectbox("Cause", options=selected, index=0, key="gc_cause")
    with c2:
        effect_opts = [f for f in selected if f != cause] or selected
        effect = st.selectbox("Effect", options=effect_opts, index=0, key="gc_effect")
    with c3:
        max_lag_gc = st.number_input("Max lag", min_value=1, max_value=30, value=7, step=1)

    pair = unshifted[[effect, cause]].dropna()
    if len(pair) < 30:
        st.warning(f"Only {len(pair)} overlapping rows — need at least 30.")
    elif cause == effect:
        st.warning("Pick two distinct features.")
    else:
        try:
            with st.spinner("Running Granger test…"):
                res = grangercausalitytests(pair.values, maxlag=int(max_lag_gc), verbose=False)
            gc_rows = []
            for lag, (stats, _models) in res.items():
                gc_rows.append(
                    {
                        "lag": lag,
                        "F": stats["ssr_ftest"][0],
                        "p_value": stats["ssr_ftest"][1],
                    }
                )
            gc_df = pd.DataFrame(gc_rows)
            st.caption(
                f"H0: `{cause}` does NOT Granger-cause `{effect}`. Reject when p < 0.05."
            )
            st.dataframe(gc_df, use_container_width=True)
            fig_gc = px.bar(
                gc_df, x="lag", y="p_value",
                labels={"p_value": "p (ssr F-test)"},
            )
            fig_gc.add_hline(y=0.05, line_dash="dash", line_color="red",
                             annotation_text="α=0.05", annotation_position="top right")
            fig_gc.update_layout(height=300)
            st.plotly_chart(fig_gc, use_container_width=True)
        except Exception as exc:
            st.error(f"Granger test failed: {exc}")
