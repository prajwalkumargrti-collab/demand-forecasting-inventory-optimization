"""Inventory Risk Assessment page for the dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.inventory import calculate_inventory_metrics, get_stockout_risk


def render(df: pd.DataFrame) -> None:
    """Render the Inventory Risk Assessment page."""
    st.title("📦 Inventory Risk Assessment")
    st.markdown("---")

    try:
        inventory_df = calculate_inventory_metrics(df)
        inventory_df = get_stockout_risk(inventory_df)

        # ── KPI Row ───────────────────────────────────────────────────
        reorder_count = int(
            (inventory_df["Stock_Status"] == "REORDER NOW").sum()
        )
        avg_days_stockout = inventory_df["Days_Until_Stockout"].mean()
        sufficient_count = int(
            (inventory_df["Stock_Status"] == "SUFFICIENT").sum()
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("🚨 Items at Risk (Reorder Now)", f"{reorder_count}")
        k2.metric("📅 Avg Days Until Stockout", f"{avg_days_stockout:.1f}")
        k3.metric("✅ Sufficient Stock", f"{sufficient_count}")

        st.markdown("---")

        # ── Risk Distribution by Store ─────────────────────────────────
        st.subheader("Stock Status Distribution by Store")

        status_counts = (
            inventory_df.groupby(["Store ID", "Stock_Status"])
            .size()
            .reset_index(name="Count")
        )
        status_order = ["REORDER NOW", "LOW STOCK", "SUFFICIENT"]
        status_colors = {
            "REORDER NOW": "#e74c3c",
            "LOW STOCK": "#f39c12",
            "SUFFICIENT": "#2ecc71",
        }

        fig_status = px.bar(
            status_counts,
            x="Store ID",
            y="Count",
            color="Stock_Status",
            barmode="group",
            title="Stock Status Count by Store",
            template="plotly_white",
            category_orders={"Stock_Status": status_order},
            color_discrete_map=status_colors,
        )
        fig_status.update_layout(
            xaxis_title="Store",
            yaxis_title="Number of Products",
            legend_title="Status",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_status, use_container_width=True)

        # ── Stockout Risk Heatmap ──────────────────────────────────────
        st.subheader("Stockout Risk Heatmap (Store × Product)")

        risk_pivot = inventory_df.pivot_table(
            index="Store ID",
            columns="Product ID",
            values="Risk_Score",
            aggfunc="mean",
        ).fillna(0)

        fig_risk = px.imshow(
            risk_pivot,
            title="Stockout Risk Score — Store × Product",
            template="plotly_white",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            aspect="auto",
            labels=dict(x="Product ID", y="Store ID", color="Risk Score"),
        )
        fig_risk.update_layout(margin=dict(t=50, b=40))
        st.plotly_chart(fig_risk, use_container_width=True)

        # ── Detailed Inventory Metrics Table ───────────────────────────
        st.subheader("Detailed Inventory Metrics")

        display_cols = [
            "Store ID",
            "Product ID",
            "Current_Inventory",
            "Estimated_Daily_Demand",
            "Safety_Stock",
            "Reorder_Point",
            "Target_Stock",
            "Recommended_Order_Quantity",
            "Stock_Status",
            "Days_Until_Stockout",
            "Risk_Score",
        ]
        available_cols = [c for c in display_cols if c in inventory_df.columns]
        display_df = inventory_df[available_cols].copy()

        def color_status(val):
            color_map = {
                "REORDER NOW": "background-color: #fadbd8; color: #c0392b; font-weight: bold",
                "LOW STOCK": "background-color: #fdebd0; color: #e67e22; font-weight: bold",
                "SUFFICIENT": "background-color: #d5f5e3; color: #27ae60; font-weight: bold",
            }
            return color_map.get(val, "")

        if hasattr(display_df.style, "map"):
            styled = display_df.style.map(color_status, subset=["Stock_Status"])
        else:
            styled = display_df.style.applymap(color_status, subset=["Stock_Status"])
        st.dataframe(styled, use_container_width=True, height=450)

        # ── Replenishment Summary ──────────────────────────────────────
        st.subheader("Recommended Order Quantity by Product")

        replenish = (
            inventory_df.groupby(["Product ID", "Store ID"])[
                "Recommended_Order_Quantity"
            ]
            .sum()
            .reset_index()
        )

        fig_replenish = px.bar(
            replenish,
            x="Product ID",
            y="Recommended_Order_Quantity",
            color="Store ID",
            title="Recommended Replenishment Orders (Stacked by Store)",
            template="plotly_white",
            barmode="stack",
        )
        fig_replenish.update_layout(
            xaxis_title="Product",
            yaxis_title="Order Quantity",
            legend_title="Store",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_replenish, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering Inventory Risk Assessment: {e}")
