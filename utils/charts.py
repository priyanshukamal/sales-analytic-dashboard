"""
utils/charts.py
───────────────
All Plotly chart factory functions.
Each returns a go.Figure ready for st.plotly_chart().
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Brand colours ─────────────────────────────────────────
PALETTE  = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6", "#8b5cf6"]
BG       = "rgba(0,0,0,0)"
FONT_CLR = "#e2e8f0"
GRID_CLR = "rgba(255,255,255,0.07)"

_layout_defaults = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=FONT_CLR, family="Inter, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(gridcolor=GRID_CLR, zeroline=False),
    yaxis=dict(gridcolor=GRID_CLR, zeroline=False),
)


def _apply(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_layout_defaults)
    return fig


# ─────────────────────────────────────────────────────────────
# 1. MONTHLY REVENUE LINE CHART
# ─────────────────────────────────────────────────────────────
def monthly_revenue_chart(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_df["period"],
        y=monthly_df["revenue"],
        mode="lines+markers",
        name="Revenue",
        line=dict(color="#6366f1", width=3),
        marker=dict(size=6, color="#6366f1"),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.12)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="Monthly Revenue Trend", **_layout_defaults)
    return fig


# ─────────────────────────────────────────────────────────────
# 2. CATEGORY BAR CHART
# ─────────────────────────────────────────────────────────────
def category_bar_chart(cat_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=cat_df["category"],
        y=cat_df["revenue"],
        marker=dict(
            color=PALETTE[:len(cat_df)],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="Revenue by Category", **_layout_defaults)
    return fig


# ─────────────────────────────────────────────────────────────
# 3. REGION CHOROPLETH-STYLE HORIZONTAL BAR
# ─────────────────────────────────────────────────────────────
def region_bar_chart(reg_df: pd.DataFrame) -> go.Figure:
    reg_sorted = reg_df.sort_values("revenue")
    fig = go.Figure(go.Bar(
        x=reg_sorted["revenue"],
        y=reg_sorted["region"],
        orientation="h",
        marker=dict(
            color=reg_sorted["revenue"],
            colorscale="Viridis",
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="Revenue by Region", **_layout_defaults)
    return fig


# ─────────────────────────────────────────────────────────────
# 4. CHANNEL DONUT CHART
# ─────────────────────────────────────────────────────────────
def channel_donut(ch_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=ch_df["channel"],
        values=ch_df["revenue"],
        hole=0.55,
        marker=dict(colors=PALETTE),
        hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
        textinfo="label+percent",
        textfont=dict(color=FONT_CLR),
    ))
    fig.update_layout(
        title="Sales Channel Distribution",
        paper_bgcolor=BG,
        font=dict(color=FONT_CLR, family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 5. SCATTER: REVENUE vs PROFIT (with category colour)
# ─────────────────────────────────────────────────────────────
def revenue_profit_scatter(df: pd.DataFrame) -> go.Figure:
    # Sample for performance
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="revenue",
        y="profit",
        color="category",
        size="quantity",
        color_discrete_sequence=PALETTE,
        hover_data={"revenue": ":,.0f", "profit": ":,.0f", "category": True},
        labels={"revenue": "Revenue (₹)", "profit": "Profit (₹)"},
    )
    fig.update_layout(title="Revenue vs Profit by Category", **_layout_defaults)
    return fig


# ─────────────────────────────────────────────────────────────
# 6. QUARTERLY GROUPED BAR (Revenue & Profit)
# ─────────────────────────────────────────────────────────────
def quarterly_grouped_bar(q_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=q_df["label"], y=q_df["revenue"],
        name="Revenue", marker_color="#6366f1",
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=q_df["label"], y=q_df["profit"],
        name="Profit", marker_color="#10b981",
        hovertemplate="<b>%{x}</b><br>Profit: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Quarterly Revenue & Profit",
        barmode="group",
        **_layout_defaults,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 7. HEATMAP: Category × Month revenue
# ─────────────────────────────────────────────────────────────
def category_month_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["category", "month_label"])["revenue"]
        .sum()
        .unstack(fill_value=0)
    )
    # Month order
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Purples",
        hovertemplate="<b>%{y} – %{x}</b><br>Revenue: ₹%{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="Revenue Heatmap: Category × Month", **_layout_defaults)
    return fig


# ─────────────────────────────────────────────────────────────
# 8. SEGMENT RADAR CHART
# ─────────────────────────────────────────────────────────────
def segment_radar(seg_df: pd.DataFrame) -> go.Figure:
    metrics = ["revenue", "profit", "orders", "avg_order"]
    labels  = ["Revenue", "Profit", "Orders", "Avg Order"]

    # Normalise 0-1 per metric
    norm = seg_df[metrics].copy()
    for col in metrics:
        rng = norm[col].max() - norm[col].min()
        norm[col] = (norm[col] - norm[col].min()) / rng if rng else 0.5

    fig = go.Figure()
    for i, row in seg_df.iterrows():
        vals = [norm.loc[i, m] for m in metrics]
        vals.append(vals[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels + [labels[0]],
            fill="toself",
            name=row["customer_segment"],
            line_color=PALETTE[i % len(PALETTE)],
            opacity=0.7,
        ))
    fig.update_layout(
        title="Customer Segment Comparison",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor=GRID_CLR),
            angularaxis=dict(gridcolor=GRID_CLR),
        ),
        paper_bgcolor=BG,
        font=dict(color=FONT_CLR, family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
