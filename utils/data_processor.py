"""
utils/data_processor.py
───────────────────────
Handles all data loading, cleaning, and feature engineering.
Called once at app startup and cached by Streamlit.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
def load_data(filepath: str = "data/sales_data.csv") -> pd.DataFrame:
    """Load CSV and apply correct dtypes."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


# ─────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Perform data quality fixes.
    Returns cleaned DataFrame + a report dict for the UI.
    """
    report = {}

    # ── 2a. Track initial shape ────────────────────────────
    report["rows_before"] = len(df)

    # ── 2b. Drop full duplicates ───────────────────────────
    before_dup = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before_dup - len(df)

    # ── 2c. Fix negative quantities (data-entry errors) ────
    neg_mask = df["quantity"] < 0
    report["negative_qty_fixed"] = int(neg_mask.sum())
    df.loc[neg_mask, "quantity"] = df.loc[neg_mask, "quantity"].abs()

    # ── 2d. Impute missing price with category median ──────
    missing_price = df["price"].isna().sum()
    report["missing_price_imputed"] = int(missing_price)
    df["price"] = df.groupby("category")["price"].transform(
        lambda s: s.fillna(s.median())
    )

    # ── 2e. Recalculate revenue & profit after fixes ───────
    df["revenue"] = (
        df["price"] * df["quantity"] * (1 - df["discount_pct"] / 100)
    ).round(2)
    df["profit"] = (df["revenue"] * 0.22).round(2)  # normalised margin

    # ── 2f. Final shape ────────────────────────────────────
    report["rows_after"] = len(df)

    return df, report


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time-based and business features."""

    # Time columns
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week"]    = df["date"].dt.isocalendar().week.astype(int)
    df["day_name"]= df["date"].dt.day_name()

    # Month label (Jan, Feb …) preserving sort order
    df["month_label"] = df["date"].dt.strftime("%b")

    # Revenue bucket
    df["revenue_bucket"] = pd.cut(
        df["revenue"],
        bins=[0, 100, 500, 1000, 5000, np.inf],
        labels=["<100", "100-500", "500-1k", "1k-5k", "5k+"],
    )

    # Profit margin %
    df["margin_pct"] = (df["profit"] / df["revenue"].replace(0, np.nan) * 100).round(1)

    return df


# ─────────────────────────────────────────────────────────────
# 4. ORCHESTRATOR
# ─────────────────────────────────────────────────────────────
def get_clean_df(filepath: str = "data/sales_data.csv"):
    """Single entry-point used by Streamlit."""
    raw        = load_data(filepath)
    cleaned, report = clean_data(raw)
    final      = engineer_features(cleaned)
    return final, report
