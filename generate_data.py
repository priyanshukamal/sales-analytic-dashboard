"""
generate_data.py
Generates a realistic retail sales dataset for the dashboard demo.
Run once: python generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────
N_ROWS      = 5_000
START_DATE  = datetime(2022, 1, 1)
END_DATE    = datetime(2024, 12, 31)

CATEGORIES  = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Beauty"]
REGIONS     = ["North", "South", "East", "West", "Central"]
CHANNELS    = ["Online", "In-Store", "Mobile App"]
SEGMENTS    = ["Premium", "Standard", "Budget"]

# Category-level base prices
PRICE_RANGES = {
    "Electronics":    (150, 1200),
    "Clothing":       (20,  200),
    "Home & Garden":  (15,  500),
    "Sports":         (25,  400),
    "Books":          (8,   60),
    "Beauty":         (10,  150),
}

# ── Generate rows ─────────────────────────────────────────
date_range = (END_DATE - START_DATE).days
dates      = [START_DATE + timedelta(days=random.randint(0, date_range)) for _ in range(N_ROWS)]
categories = random.choices(CATEGORIES, k=N_ROWS)

records = []
for i, (date, cat) in enumerate(zip(dates, categories)):
    lo, hi   = PRICE_RANGES[cat]
    price    = round(random.uniform(lo, hi), 2)
    quantity = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 12, 8, 5])[0]
    discount = random.choices([0, 5, 10, 15, 20], weights=[40, 25, 20, 10, 5])[0]
    revenue  = round(price * quantity * (1 - discount / 100), 2)
    profit   = round(revenue * random.uniform(0.10, 0.35), 2)

    # Inject some missing / dirty data for cleaning demo
    if random.random() < 0.02:
        price = None          # 2 % missing price
    if random.random() < 0.01:
        quantity = -quantity  # 1 % negative quantity (data entry error)

    records.append({
        "order_id":        f"ORD-{10000 + i}",
        "date":            date.strftime("%Y-%m-%d"),
        "category":        cat,
        "region":          random.choice(REGIONS),
        "channel":         random.choice(CHANNELS),
        "customer_segment":random.choice(SEGMENTS),
        "price":           price,
        "quantity":        quantity,
        "discount_pct":    discount,
        "revenue":         revenue,
        "profit":          profit,
        "customer_id":     f"CUST-{random.randint(1000, 3000)}",
    })

df = pd.DataFrame(records)
df.to_csv("data/sales_data.csv", index=False)
print(f"✅  Generated {len(df)} rows → data/sales_data.csv")
print(df.head())
