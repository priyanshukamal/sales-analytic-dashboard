"""
utils/eda.py
────────────
All analytical computations (KPIs, trends, segment analysis).
Pure pandas — no Streamlit imports here, so it stays testable.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    """Top-line business metrics."""
    total_revenue  = df["revenue"].sum()
    total_profit   = df["profit"].sum()
    total_orders   = len(df)
    total_customers= df["customer_id"].nunique()
    avg_order_val  = df["revenue"].mean()
    avg_margin     = (total_profit / total_revenue * 100) if total_revenue else 0

    # Month-over-month growth (latest vs previous month)
    monthly = df.groupby(["year", "month"])["revenue"].sum().reset_index()
    monthly = monthly.sort_values(["year", "month"])
    if len(monthly) >= 2:
        last_rev  = monthly["revenue"].iloc[-1]
        prev_rev  = monthly["revenue"].iloc[-2]
        mom_growth = ((last_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
    else:
        mom_growth = 0

    return {
        "total_revenue":   round(total_revenue, 2),
        "total_profit":    round(total_profit, 2),
        "total_orders":    total_orders,
        "unique_customers":total_customers,
        "avg_order_value": round(avg_order_val, 2),
        "avg_margin_pct":  round(avg_margin, 1),
        "mom_growth_pct":  round(mom_growth, 1),
    }


# ─────────────────────────────────────────────────────────────
# TIME SERIES
# ─────────────────────────────────────────────────────────────
def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue aggregated with period label."""
    result = (
        df.groupby(["year", "month"])
        .agg(revenue=("revenue", "sum"), orders=("order_id", "count"))
        .reset_index()
        .sort_values(["year", "month"])
    )
    result["period"] = pd.to_datetime(
        result[["year", "month"]].assign(day=1)
    )
    return result


def quarterly_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["year", "quarter"])
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
        .reset_index()
        .sort_values(["year", "quarter"])
        .assign(label=lambda x: x["year"].astype(str) + " Q" + x["quarter"].astype(str))
    )


# ─────────────────────────────────────────────────────────────
# CATEGORY ANALYSIS
# ─────────────────────────────────────────────────────────────
def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "count"),
            avg_order=("revenue", "mean"),
        )
        .reset_index()
        .assign(margin_pct=lambda x: (x["profit"] / x["revenue"] * 100).round(1))
        .sort_values("revenue", ascending=False)
    )


def region_performance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("region")
        .agg(revenue=("revenue", "sum"), orders=("order_id", "count"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )


def channel_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("channel")
        .agg(revenue=("revenue", "sum"), orders=("order_id", "count"))
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────
# CUSTOMER SEGMENT
# ─────────────────────────────────────────────────────────────
def segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("customer_segment")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "count"),
            avg_order=("revenue", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )


# ─────────────────────────────────────────────────────────────
# AI INSIGHTS  (pattern-based rules, phrased like ML output)
# ─────────────────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame, kpis: dict) -> list[str]:
    """
    Auto-generate bullet insights from the data.
    Can be swapped for an LLM call in a future version.
    """
    insights = []

    # Top category
    cat_perf = category_performance(df)
    top_cat  = cat_perf.iloc[0]
    insights.append(
        f"📦 **{top_cat['category']}** is the highest-revenue category "
        f"(₹{top_cat['revenue']:,.0f}), contributing "
        f"{top_cat['revenue']/kpis['total_revenue']*100:.1f}% of total revenue."
    )

    # Best margin category
    best_margin = cat_perf.loc[cat_perf["margin_pct"].idxmax()]
    insights.append(
        f"💰 **{best_margin['category']}** has the best profit margin "
        f"({best_margin['margin_pct']:.1f}%), making it the most profitable category."
    )

    # MoM growth signal
    mom = kpis["mom_growth_pct"]
    if mom > 0:
        insights.append(
            f"📈 Revenue grew **{mom:.1f}%** month-over-month — positive momentum detected."
        )
    else:
        insights.append(
            f"⚠️ Revenue declined **{abs(mom):.1f}%** month-over-month — worth investigating."
        )

    # Top region
    reg = region_performance(df).iloc[0]
    insights.append(
        f"🗺️ **{reg['region']}** region leads with ₹{reg['revenue']:,.0f} "
        f"across {reg['orders']:,} orders."
    )

    # Channel insight
    ch = channel_breakdown(df).sort_values("revenue", ascending=False).iloc[0]
    insights.append(
        f"📱 **{ch['channel']}** is the top sales channel by revenue "
        f"(₹{ch['revenue']:,.0f})."
    )

    # Discount impact
    high_disc = df[df["discount_pct"] >= 15]
    if len(high_disc) > 0:
        disc_rev_pct = high_disc["revenue"].sum() / df["revenue"].sum() * 100
        insights.append(
            f"🏷️ Orders with ≥15% discount account for "
            f"**{disc_rev_pct:.1f}%** of revenue — monitor margin erosion."
        )

    return insights
