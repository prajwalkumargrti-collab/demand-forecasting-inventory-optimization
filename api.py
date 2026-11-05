"""
FastAPI endpoint for demand forecasting.

Accepts a product/store and forecast horizon and returns validated forecast output.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np

from src.config import DATA_PATH, LEAD_TIME_DAYS, REVIEW_PERIOD_DAYS, SERVICE_LEVEL_Z, LOOKBACK_DAYS
from src.data_loader import load_raw_data, preprocess_data
from src.forecasting import prepare_features, train_model, generate_forecast, get_model_metrics
from src.inventory import calculate_inventory_metrics, get_stockout_risk

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Demand Forecasting & Inventory Optimization API",
    description="API for generating demand forecasts and inventory recommendations.",
    version="1.0.0",
)

# ── Global state (loaded once at startup) ────────────────────────────────────
_state = {}


@app.on_event("startup")
def startup_event():
    """Load data and train model at startup."""
    try:
        raw_df = load_raw_data(DATA_PATH)
        processed_df = preprocess_data(raw_df)
        model, test_preds, test_actuals, feat_imp, metrics, baseline = train_model(processed_df)

        _state["df"] = processed_df
        _state["model"] = model
        _state["metrics"] = metrics
        _state["baseline_metrics"] = baseline

        print("✅ API started — data loaded and model trained.")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise


# ── Response models ──────────────────────────────────────────────────────────

class ForecastItem(BaseModel):
    date: str
    store_id: str
    product_id: str
    forecast_demand: float


class ForecastResponse(BaseModel):
    store_id: str
    product_id: str
    horizon_days: int
    forecasts: List[ForecastItem]
    model_mae: float
    model_rmse: float


class InventoryItem(BaseModel):
    store_id: str
    product_id: str
    current_inventory: int
    estimated_daily_demand: float
    safety_stock: int
    reorder_point: int
    target_stock: int
    recommended_order_quantity: int
    stock_status: str
    days_until_stockout: float
    risk_score: float


class InventoryResponse(BaseModel):
    items: List[InventoryItem]


class HealthResponse(BaseModel):
    status: str
    data_rows: int
    model_mae: float
    stores: List[str]
    products: List[str]


# ── Validation helpers ───────────────────────────────────────────────────────

def _validate_store(store_id: str):
    valid = sorted(_state["df"]["Store ID"].unique())
    if store_id not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Store ID '{store_id}'. Valid values: {valid}",
        )


def _validate_product(product_id: str):
    valid = sorted(_state["df"]["Product ID"].unique())
    if product_id not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Product ID '{product_id}'. Valid values: {valid}",
        )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health and basic info."""
    df = _state["df"]
    return HealthResponse(
        status="healthy",
        data_rows=len(df),
        model_mae=round(_state["metrics"]["MAE"], 2),
        stores=sorted(df["Store ID"].unique().tolist()),
        products=sorted(df["Product ID"].unique().tolist()),
    )


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    store_id: str = Query(..., description="Store ID, e.g. S001"),
    product_id: str = Query(..., description="Product ID, e.g. P0001"),
    horizon: int = Query(30, ge=1, le=90, description="Forecast horizon in days (1-90)"),
):
    """
    Generate demand forecast for a specific store/product combination.

    Returns forecasted demand for the requested horizon along with model metrics.
    """
    _validate_store(store_id)
    _validate_product(product_id)

    df = _state["df"]
    model = _state["model"]

    try:
        forecast_df = generate_forecast(model, df, horizon_days=horizon)

        # Filter for requested store/product
        mask = (forecast_df["Store ID"] == store_id) & (forecast_df["Product ID"] == product_id)
        result = forecast_df[mask].copy()

        if result.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast generated for Store={store_id}, Product={product_id}. "
                       "This may happen if there is insufficient historical data.",
            )

        forecasts = [
            ForecastItem(
                date=row["Date"].strftime("%Y-%m-%d"),
                store_id=store_id,
                product_id=product_id,
                forecast_demand=round(float(row["Forecast_Demand"]), 2),
            )
            for _, row in result.iterrows()
        ]

        return ForecastResponse(
            store_id=store_id,
            product_id=product_id,
            horizon_days=horizon,
            forecasts=forecasts,
            model_mae=round(_state["metrics"]["MAE"], 2),
            model_rmse=round(_state["metrics"]["RMSE"], 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


@app.get("/inventory", response_model=InventoryResponse)
def get_inventory(
    store_id: Optional[str] = Query(None, description="Filter by Store ID"),
    product_id: Optional[str] = Query(None, description="Filter by Product ID"),
):
    """
    Get inventory metrics and replenishment recommendations.

    Optionally filter by store and/or product.
    """
    if store_id:
        _validate_store(store_id)
    if product_id:
        _validate_product(product_id)

    df = _state["df"]

    try:
        inv_df = calculate_inventory_metrics(df)
        inv_df = get_stockout_risk(inv_df)

        if store_id:
            inv_df = inv_df[inv_df["Store ID"] == store_id]
        if product_id:
            inv_df = inv_df[inv_df["Product ID"] == product_id]

        if inv_df.empty:
            raise HTTPException(
                status_code=404,
                detail="No inventory data found for the given filters.",
            )

        items = [
            InventoryItem(
                store_id=row["Store ID"],
                product_id=row["Product ID"],
                current_inventory=int(row["Current_Inventory"]),
                estimated_daily_demand=round(float(row["Estimated_Daily_Demand"]), 2),
                safety_stock=int(row["Safety_Stock"]),
                reorder_point=int(row["Reorder_Point"]),
                target_stock=int(row["Target_Stock"]),
                recommended_order_quantity=int(row["Recommended_Order_Quantity"]),
                stock_status=row["Stock_Status"],
                days_until_stockout=round(float(row["Days_Until_Stockout"]), 1),
                risk_score=round(float(row["Risk_Score"]), 1),
            )
            for _, row in inv_df.iterrows()
        ]

        return InventoryResponse(items=items)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory calculation failed: {str(e)}")


@app.get("/stores")
def list_stores():
    """List all available store IDs."""
    return sorted(_state["df"]["Store ID"].unique().tolist())


@app.get("/products")
def list_products():
    """List all available product IDs."""
    return sorted(_state["df"]["Product ID"].unique().tolist())

