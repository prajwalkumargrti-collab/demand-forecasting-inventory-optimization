"""Demand Overview page for the dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render the Demand Overview page."""
    st.title("📈 Demand Overview")
    st.markdown("---")

    try:
        # ── KPI Cards ──────────────────────────────────────────────────
        total_demand = filtered_df["Demand"].sum()
        avg_daily_demand = filtered_df.groupby("Date")["Demand"].sum().mean()
        total_units_sold = filtered_df["Units Sold"].sum()
        forecast_error_mae = (filtered_df["Demand"] - filtered_df["Units Sold"]).abs().mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Demand", f"{total_demand:,.0f}")
        k2.metric("Avg Daily Demand", f"{avg_daily_demand:,.1f}")
        k3.metric("Total Units Sold", f"{total_units_sold:,.0f}")
        k4.metric("Forecast Error (MAE)", f"{forecast_error_mae:,.2f}")

        st.markdown("---")

        # ── Actual Demand Over Time ────────────────────────────────────
        daily_demand = (
            filtered_df.groupby("Date")["Demand"]
            .sum()
            .reset_index()
            .sort_values("Date")
        )

        fig_line = px.line(
            daily_demand,
            x="Date",
            y="Demand",
            title="Daily Aggregate Demand Over Time",
            template="plotly_white",
        )
        fig_line.update_traces(line=dict(color="#1f4e79", width=2))
        fig_line.update_layout(
            xaxis_title="Date",
            yaxis_title="Demand",
            hovermode="x unified",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Demand by Category ─────────────────────────────────────────
        cat_demand = (
            filtered_df.groupby("Category")["Demand"]
            .sum()
            .reset_index()
            .sort_values("Demand", ascending=True)
        )

        fig_bar = px.bar(
            cat_demand,
            x="Demand",
            y="Category",
            orientation="h",
            color="Category",
            title="Total Demand by Category",
            template="plotly_white",
        )
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="Total Demand",
            yaxis_title="",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Demand by Region & Season ──────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            region_demand = (
                filtered_df.groupby("Region")["Demand"].sum().reset_index()
            )
            fig_pie = px.pie(
                region_demand,
                values="Demand",
                names="Region",
                title="Demand by Region",
                template="plotly_white",
                hole=0.35,
            )
            fig_pie.update_layout(margin=dict(t=50, b=40))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            df_season = filtered_df.copy()
            df_season["Month"] = pd.to_datetime(df_season["Date"]).dt.month
            conditions = [
                df_season["Month"].isin([12, 1, 2]),
                df_season["Month"].isin([3, 4, 5]),
                df_season["Month"].isin([6, 7, 8]),
                df_season["Month"].isin([9, 10, 11]),
            ]
            choices = ["Winter", "Spring", "Summer", "Fall"]
            df_season["Season"] = np.select(conditions, choices, default="Unknown")

            season_demand = (
                df_season.groupby("Season")["Demand"].sum().reset_index()
            )
            season_order = ["Winter", "Spring", "Summer", "Fall"]
            season_demand["Season"] = pd.Categorical(
                season_demand["Season"], categories=season_order, ordered=True
            )
            season_demand = season_demand.sort_values("Season")

            fig_season = px.bar(
                season_demand,
                x="Season",
                y="Demand",
                title="Demand by Season",
                template="plotly_white",
                color="Season",
            )
            fig_season.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Total Demand",
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_season, use_container_width=True)

        # ── Heatmap: Average Demand by Store × Product ─────────────────
        st.subheader("Average Demand by Store × Product")
        heatmap_data = (
            filtered_df.groupby(["Store ID", "Product ID"])["Demand"]
            .mean()
            .reset_index()
        )
        heatmap_pivot = heatmap_data.pivot(
            index="Store ID", columns="Product ID", values="Demand"
        )
        heatmap_pivot = heatmap_pivot.fillna(0)

        fig_heat = px.imshow(
            heatmap_pivot,
            title="Average Demand — Store × Product",
            template="plotly_white",
            color_continuous_scale="Blues",
            aspect="auto",
            labels=dict(x="Product ID", y="Store ID", color="Avg Demand"),
        )
        fig_heat.update_layout(margin=dict(t=50, b=40))
        st.plotly_chart(fig_heat, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering Demand Overview: {e}")
