"""
Demand Forecasting & Inventory Optimization Dashboard
======================================================
Main entry-point for the Streamlit multi-page application.
"""

import streamlit as st
import pandas as pd

from src.config import APP_TITLE, DATA_PATH
from src.data_loader import load_raw_data, preprocess_data, get_filtered_data
from src.forecasting import train_model

from pages import demand_overview, product_store_forecasts, inventory_risk, model_performance

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    """Load and preprocess the raw dataset (cached)."""
    raw = load_raw_data(DATA_PATH)
    return preprocess_data(raw)


@st.cache_resource(show_spinner="Training forecasting model…")
def cached_train_model(_df: pd.DataFrame):
    """Train the model once and cache the resource."""
    model, test_preds, test_actuals, feature_importances, metrics, baseline_metrics = train_model(_df)
    return model, test_preds, test_actuals, feature_importances, metrics, baseline_metrics


# ── Load data & model ─────────────────────────────────────────────────────
try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

try:
    model, test_preds, test_actuals, feature_importances, metrics, baseline_metrics = cached_train_model(df)
except Exception as e:
    st.error(f"Failed to train model: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"📊 {APP_TITLE}")
    page = st.radio(
        "Navigation",
        options=[
            "Demand Overview",
            "Product/Store Forecasts",
            "Inventory Risk",
            "Model Performance",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("🔍 Filters")

    all_stores = sorted(df["Store ID"].unique().tolist())
    selected_stores = st.multiselect("Store", all_stores, default=all_stores)

    all_categories = sorted(df["Category"].unique().tolist())
    selected_categories = st.multiselect("Category", all_categories, default=all_categories)

    min_date = pd.to_datetime(df["Date"]).min().date()
    max_date = pd.to_datetime(df["Date"]).max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown(
            """
            **Demand Forecasting & Inventory Optimization**

            This dashboard provides end-to-end visibility into demand
            patterns, product-level forecasts, inventory health, and
            model performance across 5 stores and 20 products
            (2022-01 → 2024-01).

            Built with **Streamlit**, **Plotly**, and machine-learning
            forecasting models.
            """
        )

# ── Apply filters ──────────────────────────────────────────────────────────
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_df = get_filtered_data(
    df,
    stores=selected_stores if selected_stores else all_stores,
    products=None,
    categories=selected_categories if selected_categories else all_categories,
    date_range=(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    ),
)

if filtered_df.empty:
    st.warning("No data matches the current filters. Please adjust your selections.")
    st.stop()

# ── Page routing ───────────────────────────────────────────────────────────
if page == "Demand Overview":
    demand_overview.render(df, filtered_df)
elif page == "Product/Store Forecasts":
    product_store_forecasts.render(df, model, test_preds, test_actuals)
elif page == "Inventory Risk":
    inventory_risk.render(filtered_df)
elif page == "Model Performance":
    model_performance.render(
        metrics, baseline_metrics, test_actuals, test_preds, feature_importances, df
    )
