"""
Inventory optimisation calculations — safety stock, reorder points, and
stockout-risk scoring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    LEAD_TIME_DAYS,
    LOOKBACK_DAYS,
    REVIEW_PERIOD_DAYS,
    SERVICE_LEVEL_Z,
    STOCK_STATUS_LOW,
    STOCK_STATUS_OK,
    STOCK_STATUS_REORDER,
)


# ══════════════════════════════════════════════════════════════════════════════
# Core inventory metrics
# ══════════════════════════════════════════════════════════════════════════════

def calculate_inventory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute inventory KPIs for every (Store ID, Product ID) combination.

    For each combination the **last** ``LOOKBACK_DAYS`` days of data are
    used to estimate demand statistics.  The following columns are produced:

    * ``estimated_daily_demand`` – mean daily demand over the look-back window.
    * ``std_daily`` – standard deviation of daily demand.
    * ``current_inventory`` – the most recent ``Inventory Level`` value.
    * ``safety_stock`` – Z × σ × √(lead_time + review_period).
    * ``reorder_point`` – daily_demand × lead_time + safety_stock.
    * ``target_stock`` – daily_demand × (lead_time + review_period) + safety_stock.
    * ``recommended_order_qty`` – max(0, target_stock − current_inventory).
    * ``stock_status`` – one of ``REORDER NOW``, ``LOW STOCK``, ``SUFFICIENT``.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed sales dataframe with at least ``Date``, ``Store ID``,
        ``Product ID``, ``Demand``, and ``Inventory Level`` columns.

    Returns
    -------
    pd.DataFrame
        One row per (Store ID, Product ID) with the metrics listed above.
    """
    df = df.copy()
    max_date = df["Date"].max()
    lookback_start = max_date - pd.Timedelta(days=LOOKBACK_DAYS)
    recent = df[df["Date"] >= lookback_start]

    records = []
    for (store, product), grp in recent.groupby(["Store ID", "Product ID"]):
        grp = grp.sort_values("Date")

        daily_demand = grp["Demand"].mean()
        std_daily    = grp["Demand"].std(ddof=1)
        if np.isnan(std_daily):
            std_daily = 0.0
        current_inv  = grp["Inventory Level"].iloc[-1]

        safety_stock = SERVICE_LEVEL_Z * std_daily * np.sqrt(
            LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS
        )
        reorder_point = daily_demand * LEAD_TIME_DAYS + safety_stock
        target_stock  = daily_demand * (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS) + safety_stock

        recommended_order_qty = max(0.0, target_stock - current_inv)
        days_stockout = days_until_stockout(current_inv, daily_demand)

        # Determine status
        if current_inv <= reorder_point:
            status = STOCK_STATUS_REORDER
        elif current_inv <= reorder_point * 1.25:
            status = STOCK_STATUS_LOW
        else:
            status = STOCK_STATUS_OK

        records.append({
            "Store ID":                    store,
            "Product ID":                  product,
            "Current_Inventory":           int(round(current_inv)),
            "Estimated_Daily_Demand":      round(daily_demand, 2),
            "Safety_Stock":                int(round(safety_stock)),
            "Reorder_Point":               int(round(reorder_point)),
            "Target_Stock":                int(round(target_stock)),
            "Recommended_Order_Quantity":  int(round(recommended_order_qty)),
            "Stock_Status":                status,
            "Days_Until_Stockout":         round(days_stockout, 1) if not np.isinf(days_stockout) else 999.0,
            # Aliases for backwards compatibility
            "current_inventory":           round(current_inv, 2),
            "estimated_daily_demand":      round(daily_demand, 2),
            "std_daily":                   round(std_daily, 2),
            "safety_stock":                round(safety_stock, 2),
            "reorder_point":               round(reorder_point, 2),
            "target_stock":                round(target_stock, 2),
            "recommended_order_qty":       round(recommended_order_qty, 2),
            "stock_status":                status,
            "days_until_stockout":         round(days_stockout, 1) if not np.isinf(days_stockout) else 999.0,
        })

    inventory_df = pd.DataFrame(records)
    return inventory_df


# ══════════════════════════════════════════════════════════════════════════════
# Stockout risk scoring
# ══════════════════════════════════════════════════════════════════════════════

def get_stockout_risk(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``risk_score`` (0–100) column to the inventory metrics table.

    The score reflects how close current inventory is to (or below) the
    reorder point:

    * **100** – current inventory is zero or negative.
    * **75–99** – current inventory is at or below the reorder point.
    * **50–74** – between reorder point and 1.5× reorder point.
    * **0–49** – comfortably above reorder.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        Output of :func:`calculate_inventory_metrics`.

    Returns
    -------
    pd.DataFrame
        Same dataframe with an added ``risk_score`` and ``Risk_Score`` column.
    """
    df = inventory_df.copy()

    def _score(row: pd.Series) -> float:
        current = row.get("Current_Inventory", row.get("current_inventory", 0))
        rop     = row.get("Reorder_Point", row.get("reorder_point", 0))

        if current <= 0:
            return 100.0
        if rop == 0:
            return 0.0

        ratio = current / rop  # >1 means above reorder point

        if ratio <= 1.0:
            # At or below reorder point → scale 75-100
            score = 100 - 25 * ratio
        elif ratio <= 1.5:
            # Between 1× and 1.5× reorder point → scale 50-75
            score = 75 - 50 * (ratio - 1.0)
        else:
            # Comfortably above → scale 0-50
            score = max(0.0, 50 - (ratio - 1.5) * (50 / 1.5))

        return round(min(max(score, 0.0), 100.0), 1)

    df["risk_score"] = df.apply(_score, axis=1)
    df["Risk_Score"] = df["risk_score"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Simple stockout countdown
# ══════════════════════════════════════════════════════════════════════════════

def days_until_stockout(current_inv: float, daily_demand: float) -> float:
    """Estimate the number of days until inventory reaches zero.

    Parameters
    ----------
    current_inv : float
        Current on-hand inventory units.
    daily_demand : float
        Estimated average daily demand.

    Returns
    -------
    float
        Days until stockout.  Returns ``float('inf')`` when daily demand
        is zero or negative (i.e. stock will never run out at that rate).
    """
    if daily_demand <= 0:
        return float("inf")
    return current_inv / daily_demand
