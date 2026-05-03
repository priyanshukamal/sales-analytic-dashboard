"""
app.py  —  AI-Powered Sales Analytics Dashboard
─────────────────────────────────────────────────
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import date

from utils.data_processor import get_clean_df
from utils.eda import (
    compute_kpis,
    monthly_revenue_trend,
    quarterly_trend,
    category_performance,
    region_performance,
    channel_breakdown,
    segment_analysis,
    generate_insights,
)
from utils.charts import (
    monthly_revenue_chart,
    category_bar_chart,
    region_bar_chart,
    channel_donut,
    revenue_profit_scatter,
    quarterly_grouped_bar,
    category_month_heatmap,
    segment_radar,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark glass-morphism UI)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* KPI cards */
.kpi-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-value { font-size: 2rem; font-weight: 700; color: #a5b4fc; }
.kpi-label { font-size: 0.78rem; color: #94a3b8; letter-spacing: .06em; text-transform: uppercase; margin-top: 4px; }
.kpi-delta { font-size: 0.82rem; margin-top: 6px; font-weight: 500; }
.delta-up   { color: #34d399; }
.delta-down { color: #f87171; }

/* Section headers */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 28px 0 12px;
    padding-left: 8px;
    border-left: 3px solid #6366f1;
}

/* Insight card */
.insight-box {
    background: rgba(99,102,241,0.10);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* Data quality pill */
.dq-pill {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 9999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #34d399;
    margin: 3px;
}

/* Hide hamburger menu */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING  (cached for performance)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data …")
def load():
    return get_clean_df("data/sales_data.csv")


df_full, clean_report = load()


# ─────────────────────────────────────────────────────────────
# SIDEBAR  — Filters
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filters")

    # Date range
    min_date = df_full["date"].min().date()
    max_date = df_full["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Category multi-select
    all_cats = sorted(df_full["category"].unique())
    selected_cats = st.multiselect("Category", all_cats, default=all_cats)

    # Region multi-select
    all_regions = sorted(df_full["region"].unique())
    selected_regions = st.multiselect("Region", all_regions, default=all_regions)

    # Channel
    all_channels = sorted(df_full["channel"].unique())
    selected_channels = st.multiselect("Sales Channel", all_channels, default=all_channels)

    st.markdown("---")
    st.markdown("### 🧹 Data Quality Report")
    cols_dq = [
        ("Rows loaded",       clean_report["rows_before"]),
        ("Duplicates removed",clean_report["duplicates_removed"]),
        ("Neg qty fixed",     clean_report["negative_qty_fixed"]),
        ("Missing price imp.",clean_report["missing_price_imputed"]),
        ("Clean rows",        clean_report["rows_after"]),
    ]
    for label, val in cols_dq:
        st.markdown(f'<span class="dq-pill">{label}: {val:,}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built by Priyanshu Kamal · AI Analytics Dashboard")


# ─────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_d, end_d = pd.Timestamp(min_date), pd.Timestamp(max_date)

df = df_full[
    (df_full["date"] >= start_d) &
    (df_full["date"] <= end_d) &
    (df_full["category"].isin(selected_cats)) &
    (df_full["region"].isin(selected_regions)) &
    (df_full["channel"].isin(selected_channels))
]

if df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 24px">
  <h1 style="font-size:2rem; font-weight:700; color:#e2e8f0; margin:0">
    📊 AI Sales Analytics Dashboard
  </h1>
  <p style="color:#64748b; font-size:0.9rem; margin:6px 0 0">
    Real-time insights · Data-driven decisions · Built with Python & Streamlit
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
kpis = compute_kpis(df)

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

def kpi_card(col, value, label, delta=None, prefix="₹", is_pct=False):
    if isinstance(value, float):
        fmt_val = f"{prefix}{value:,.0f}" if not is_pct else f"{value:.1f}%"
    else:
        fmt_val = f"{value:,}" if prefix == "" else f"{prefix}{value:,}"

    delta_html = ""
    if delta is not None:
        cls  = "delta-up" if delta >= 0 else "delta-down"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{sign} {abs(delta):.1f}% MoM</div>'

    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value">{fmt_val}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

kpi_card(k1, kpis["total_revenue"],   "Total Revenue",     kpis["mom_growth_pct"])
kpi_card(k2, kpis["total_profit"],    "Total Profit")
kpi_card(k3, kpis["total_orders"],    "Total Orders",      prefix="")
kpi_card(k4, kpis["unique_customers"],"Unique Customers",  prefix="")
kpi_card(k5, kpis["avg_order_value"], "Avg Order Value")
kpi_card(k6, kpis["avg_margin_pct"],  "Avg Margin",        is_pct=True, prefix="")
kpi_card(k7, kpis["mom_growth_pct"],  "MoM Growth",        is_pct=True, prefix="")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ROW 2 — Revenue trend + Category bar
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Sales Trends</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    monthly_df = monthly_revenue_trend(df)
    st.plotly_chart(monthly_revenue_chart(monthly_df), use_container_width=True)

with col_right:
    cat_df = category_performance(df)
    st.plotly_chart(category_bar_chart(cat_df), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ROW 3 — Region + Channel + Quarterly
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🗺️ Geography & Channels</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.2, 1, 1.5])

with c1:
    reg_df = region_performance(df)
    st.plotly_chart(region_bar_chart(reg_df), use_container_width=True)

with c2:
    ch_df = channel_breakdown(df)
    st.plotly_chart(channel_donut(ch_df), use_container_width=True)

with c3:
    q_df = quarterly_trend(df)
    st.plotly_chart(quarterly_grouped_bar(q_df), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ROW 4 — Scatter + Heatmap
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔬 Deep Analysis</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.plotly_chart(revenue_profit_scatter(df), use_container_width=True)

with col_b:
    st.plotly_chart(category_month_heatmap(df), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ROW 5 — Segment radar + AI Insights
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🤖 AI-Generated Insights</div>', unsafe_allow_html=True)

col_rad, col_ins = st.columns([1, 1.5])

with col_rad:
    seg_df = segment_analysis(df)
    st.plotly_chart(segment_radar(seg_df), use_container_width=True)

with col_ins:
    insights = generate_insights(df, kpis)
    st.markdown("<br>", unsafe_allow_html=True)
    for insight in insights:
        st.markdown(
            f'<div class="insight-box">{insight}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# ROW 6 — Raw data explorer (collapsible)
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Data Explorer</div>', unsafe_allow_html=True)

with st.expander("View filtered raw data", expanded=False):
    cols_show = ["order_id","date","category","region","channel",
                 "customer_segment","price","quantity","discount_pct","revenue","profit"]
    st.dataframe(
        df[cols_show].sort_values("date", ascending=False).head(500),
        use_container_width=True,
        height=350,
    )
    st.caption(f"Showing top 500 of {len(df):,} filtered rows")

    # Download button
    csv_bytes = df[cols_show].to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download filtered data as CSV",
        data=csv_bytes,
        file_name="filtered_sales_data.csv",
        mime="text/csv",
    )
