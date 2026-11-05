"""
ML-based demand forecasting with Random Forest and naive-baseline comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the ML feature set from the preprocessed sales dataframe.

    For every (Store ID, Product ID) group the data is sorted by Date and
    the following features are created:

    * **Lag features** – ``lag_7``, ``lag_14``, ``lag_28`` of ``Demand``.
    * **Rolling-window features** – ``rolling_7``, ``rolling_14``,
      ``rolling_28`` (mean) and ``rolling_7_std`` (std) of ``Demand``.
    * **Numeric features kept as-is** – ``Price``, ``Discount``,
      ``Promotion``, ``Competitor Pricing``, ``Epidemic``.
    * **One-hot encoded categoricals** – ``Category``, ``Region``,
      ``Seasonality``, ``Weather Condition``.

    Rows with NaN values introduced by lag / rolling operations are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe (must contain ``Date``, ``Store ID``,
        ``Product ID``, ``Demand`` and the raw feature columns).

    Returns
    -------
    pd.DataFrame
        Feature-enriched dataframe ready for modelling.
    """
    df = df.copy()

    # Sort within each group so lags are meaningful
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    # ── Lag & rolling features (per store-product) ───────────────────────
    group = df.groupby(["Store ID", "Product ID"])["Demand"]

    df["lag_7"]  = group.shift(7)
    df["lag_14"] = group.shift(14)
    df["lag_28"] = group.shift(28)

    df["rolling_7"]     = group.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["rolling_14"]    = group.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    df["rolling_28"]    = group.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    df["rolling_7_std"] = group.transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())

    # ── One-hot encode categoricals ──────────────────────────────────────
    cat_cols = ["Category", "Region", "Seasonality", "Weather Condition"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # ── Drop rows where lag / rolling created NaNs ───────────────────────
    lag_roll_cols = [
        "lag_7", "lag_14", "lag_28",
        "rolling_7", "rolling_14", "rolling_28", "rolling_7_std",
    ]
    df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)

    return df


# ── Helper: column list used by the model ────────────────────────────────────

def _feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the ordered list of feature columns present in *df*."""
    non_features = {
        "Date", "Store ID", "Product ID", "Demand",
        "Units Sold", "Units Ordered", "Inventory Level",
        "Year", "Month", "DayOfWeek", "WeekOfYear", "IsWeekend", "Quarter",
    }
    return [c for c in df.columns if c not in non_features]


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def get_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE between true and predicted arrays.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        ``{"MAE": …, "RMSE": …, "MAPE": …}``
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Avoid division by zero in MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 0.0

    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


def get_forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean forecast bias (mean of predictions − actuals).

    A positive bias means the model over-forecasts on average.

    Parameters
    ----------
    y_true : array-like
        Actual demand values.
    y_pred : array-like
        Predicted demand values.

    Returns
    -------
    float
        Mean bias value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


# ══════════════════════════════════════════════════════════════════════════════
# Model training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Train a Random Forest regressor to predict Demand.

    The data is split temporally: the **last 30 days** of data form the test
    set and all preceding rows are used for training.  A naïve baseline
    (predict last known demand per store-product) is also evaluated so the
    user can compare model quality.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe (output of :func:`preprocess_data`).

    Returns
    -------
    dict
        Keys:

        * ``model`` – fitted ``RandomForestRegressor``
        * ``feature_columns`` – list[str] of feature names used
        * ``y_test`` – array of actual test-set demand
        * ``y_pred`` – array of RF predictions on the test set
        * ``feature_importances`` – dict mapping feature name → importance
        * ``metrics`` – dict with MAE, RMSE, MAPE for the RF model
        * ``baseline_metrics`` – dict with MAE, RMSE for the naïve baseline
        * ``test_df`` – test-set DataFrame (for downstream plotting)
    """
    feat_df = prepare_features(df)
    feat_cols = _feature_columns(feat_df)

    # ── Temporal train / test split (last 30 days → test) ────────────────
    max_date = feat_df["Date"].max()
    cutoff   = max_date - pd.Timedelta(days=30)

    train_df = feat_df[feat_df["Date"] <= cutoff].copy()
    test_df  = feat_df[feat_df["Date"] > cutoff].copy()

    X_train = train_df[feat_cols].values
    y_train = train_df["Demand"].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df["Demand"].values

    # ── Random Forest ────────────────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    model.feature_cols_ = feat_cols
    y_pred = model.predict(X_test)

    metrics = get_model_metrics(y_test, y_pred)

    # Feature importances
    importances = dict(zip(feat_cols, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    # ── Naïve baseline: last known demand per (Store ID, Product ID) ─────
    last_known = (
        train_df.sort_values("Date")
        .groupby(["Store ID", "Product ID"])["Demand"]
        .last()
        .rename("naive_pred")
    )
    test_with_naive = test_df.merge(
        last_known, on=["Store ID", "Product ID"], how="left",
    )
    naive_pred = test_with_naive["naive_pred"].fillna(y_train.mean()).values
    baseline_metrics = {
        "MAE":  round(float(mean_absolute_error(y_test, naive_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, naive_pred))), 2),
    }

    test_df_out = test_df[["Date", "Store ID", "Product ID", "Demand"]].copy()
    test_df_out["Predicted"] = y_pred
    model.test_df = test_df_out

    return model, y_pred, y_test, importances, metrics, baseline_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Future forecast generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_forecast(
    model: RandomForestRegressor,
    df: pd.DataFrame,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Produce a future demand forecast for every Store / Product combination.

    For each combination the last available row's features are used as a
    template.  The forecast rolls forward one day at a time, updating lag
    and rolling features with the model's own predictions so that the
    feature vector stays coherent over the horizon.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model returned by :func:`train_model`.
    df : pd.DataFrame
        Preprocessed dataframe (same one passed to ``train_model``).
    horizon_days : int, optional
        Number of days to forecast into the future (default 30).

    Returns
    -------
    pd.DataFrame
        Columns: ``Date``, ``Store ID``, ``Product ID``, ``Forecast_Demand``.
    """
    feat_df = prepare_features(df)
    feat_cols = getattr(model, "feature_cols_", _feature_columns(feat_df))

    records: List[Dict[str, Any]] = []

    for (store, product), grp in feat_df.groupby(["Store ID", "Product ID"]):
        grp = grp.sort_values("Date").reset_index(drop=True)
        last_date = grp["Date"].max()

        # History of demand values for rolling / lag computation
        demand_history = grp["Demand"].tolist()

        # Use the last row as a feature template (static features stay same)
        template = grp.iloc[-1].to_dict()

        for d in range(1, horizon_days + 1):
            forecast_date = last_date + pd.Timedelta(days=d)

            row = template.copy()

            # Update lag features from demand history + previous forecasts
            hist = demand_history
            row["lag_7"]  = hist[-7]  if len(hist) >= 7  else hist[0]
            row["lag_14"] = hist[-14] if len(hist) >= 14 else hist[0]
            row["lag_28"] = hist[-28] if len(hist) >= 28 else hist[0]

            recent_7  = hist[-7:]  if len(hist) >= 7  else hist
            recent_14 = hist[-14:] if len(hist) >= 14 else hist
            recent_28 = hist[-28:] if len(hist) >= 28 else hist

            row["rolling_7"]     = float(np.mean(recent_7))
            row["rolling_14"]    = float(np.mean(recent_14))
            row["rolling_28"]    = float(np.mean(recent_28))
            row["rolling_7_std"] = float(np.std(recent_7, ddof=1)) if len(recent_7) > 1 else 0.0

            # Build feature vector in the correct column order
            x = np.array([[row.get(c, 0) for c in feat_cols]])
            pred = float(model.predict(x)[0])
            pred = max(pred, 0)  # demand cannot be negative

            demand_history.append(pred)

            records.append({
                "Date":            forecast_date,
                "Store ID":        store,
                "Product ID":      product,
                "Forecast_Demand": round(pred, 1),
            })

    forecast_df = pd.DataFrame(records)
    return forecast_df
