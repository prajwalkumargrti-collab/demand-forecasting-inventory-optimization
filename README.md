# Demand Forecasting & Inventory Optimization Using Machine Learning

An end-to-end machine learning system and interactive decision-support application for retail demand forecasting and inventory replenishment optimization.

---

## 📌 Project Overview
In modern retail, balancing supply with fluctuating customer demand is critical. Overstocking incurs high holding and depreciation costs, while understocking leads to stockouts, lost revenue, and poor customer retention. 

This project delivers a production-grade decision-support prototype that combines **machine learning demand forecasting** (Random Forest Regressor with temporal split & feature engineering) with **rigorous inventory control theory** (safety stock, dynamic reorder points, target stock, and risk scoring).

---

## 👥 Team Roles & Responsibilities

| Role | Module | Focus Area |
| :--- | :--- | :--- |
| **Role 1: Data Collection & Cleaning** | `data/`, `src/data_preprocessing.py` | Data ingestion, missing value imputation, negative value checking, date normalization. |
| **Role 2: Data Analysis & Feature Engineering** | `notebook/Feature_creation.ipynb`, `src/prepare_model_data.py` | EDA, calendar/lag/rolling feature engineering, demand pattern visualization. |
| **Role 3: Machine Learning & Demand Forecasting** | `src/forecasting.py` | Lag/rolling windows, Random Forest model training, naive baseline evaluation, metrics (MAE, RMSE, MAPE). |
| **Role 4: Inventory Management** | `src/inventory.py` | Dynamic lead-time demand, safety stock (Z=1.65, 95% service level), reorder point (ROP), stockout countdown. |
| **Role 5: Dashboard Development & Testing** | `app.py`, `api.py`, `pages/` | Interactive multi-page Streamlit dashboard, Plotly visualizations, FastAPI REST service, system testing. |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[Raw Sales Data\n76,000 Records] --> B[Data Preprocessing & Cleaning\nsrc/data_loader.py]
    B --> C[Feature Engineering\nLags, Rolling Means, Calendar, One-Hot]
    
    C --> D[ML Forecasting Engine\nRandom Forest Regressor]
    C --> E[Inventory Optimization Engine\nSafety Stock, ROP, Order Qty]
    
    D --> F[Temporal Validation\nMAE: ~22.8 | Naive Baseline: ~48.6]
    D --> G[Future Forecasts\n30-Day Forward Rolling Horizon]
    
    F --> H[Streamlit Interactive Dashboard\napp.py & pages/]
    G --> H
    E --> H
    
    D --> I[FastAPI REST Service\napi.py: /forecast, /inventory]
    E --> I
```

---

## 📂 Repository Structure

```text
demand-forecasting-inventory-optimization/
├── app.py                          # Streamlit interactive dashboard entry point
├── api.py                          # FastAPI decision-support REST API
├── requirements.txt                # Unified project dependencies
├── README.md                       # Complete project documentation
├── .gitignore                      # Git ignore file
├── data/
│   ├── raw/
│   │   └── sales_data.csv          # Immutable raw dataset (76,000 rows)
│   ├── interim/                    # Interim transformed data
│   └── processed/
│       ├── cleaned_sales.csv       # Cleaned sales data
│       └── model_ready_sales.csv   # Feature-engineered dataset
├── notebook/
│   └── Feature_creation.ipynb      # Exploratory analysis & feature prototyping
├── pages/
│   ├── __init__.py
│   ├── demand_overview.py          # Page 1: Demand KPIs, trends, category/region/seasonal breakdowns
│   ├── product_store_forecasts.py  # Page 2: Store/product trajectories, actual vs forecast, 30-day outlook
│   ├── inventory_risk.py           # Page 3: Stockout risk heatmap, replenishment orders, ROP status
│   └── model_performance.py        # Page 4: Model vs baseline evaluation, residual analysis, feature importances
├── reports/
│   ├── data_dictionary.md          # Data dictionary and schema definition
│   └── data_quality_report.md      # Data quality findings and validation checks
└── src/
    ├── __init__.py
    ├── config.py                   # Central configuration, parameters, and color palette
    ├── data_loader.py              # Data loader and dynamic filter engine
    ├── data_preprocessing.py       # Raw data validation & cleaning pipeline
    ├── forecasting.py              # ML feature extraction, model training, evaluation & forward forecasting
    ├── inventory.py                # Safety stock, reorder point, target inventory & stockout scoring
    └── prepare_model_data.py       # Aggregated model data preparation
```

---

## 📊 Dashboard Pages (Role 5 Scope)

The dashboard is structured into 4 specialized decision-support views:

### 1. 📈 Demand Overview
- **KPI Metrics**: Total Demand, Daily Average Demand, Total Units Sold, Aggregate Demand-Sales MAE.
- **Visualizations**:
  - Daily Demand Time Series with unified hover inspection.
  - Total Demand by Category (horizontal bar chart).
  - Geographic Demand distribution (donut chart by Region).
  - Seasonal Demand pattern (Winter, Spring, Summer, Fall).
  - Store × Product Average Demand Heatmap.

### 2. 🔮 Product / Store Forecasts
- **Interactive Selectors**: Filter by specific Store (`S001` - `S005`) and Product (`P0001` - `P0020`).
- **Visualizations**:
  - Full historical demand trajectory.
  - **Actual vs Forecast Overlay**: Compares actual demand vs model predictions on the held-out 30-day evaluation window, with ±1.96σ confidence intervals.
  - **30-Day Forward Forecast**: Rolling multi-step recursive projection into the future with shaded confidence bounds.
  - Tabular forecast data export for procurement planning.

### 3. 📦 Inventory Risk Assessment
- **KPI Metrics**: Items Requiring Immediate Reorder (`REORDER NOW`), Average Days Until Stockout, Sufficiently Stocked Items.
- **Visualizations**:
  - Stock Status Distribution across stores (Reorder Now vs Low Stock vs Sufficient).
  - Store × Product **Stockout Risk Heatmap** (0 - 100 risk score color-coded green to red).
  - Comprehensive inventory table with conditional status highlights.
  - Recommended Replenishment Order Quantity stacked by store.

### 4. 🎯 Model Performance
- **Validation Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE).
- **Benchmark Comparison**: Random Forest vs Naive Persistence Baseline (demonstrating >50% error reduction).
- **Diagnostic Charts**:
  - Actual vs Predicted scatter plot with 45° ideal prediction line.
  - Residual Error Distribution histogram (centering around 0).
  - Forecast Bias indicator (calibration audit).
  - Top 15 Feature Importances horizontal bar chart.
  - Absolute Error distribution by Product Category (box plot).

---

## 📦 Inventory Optimization Formulae & Assumptions

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **Lead Time ($L$)** | 3 days | Time required from order placement to warehouse arrival. |
| **Review Period ($R$)** | 7 days | Cycle time between scheduled stock evaluations. |
| **Service Level Factor ($Z$)** | 1.65 | Standard normal score corresponding to 95% cycle service level. |
| **Demand Lookback** | 28 days | Historical window used to estimate daily demand mean ($\mu_d$) and standard deviation ($\sigma_d$). |

### Formulation:
1. **Safety Stock ($SS$)**:
   $$SS = Z \times \sigma_d \times \sqrt{L + R}$$
2. **Reorder Point ($ROP$)**:
   $$ROP = (\mu_d \times L) + SS$$
3. **Target Stock ($S_{target}$)**:
   $$S_{target} = (\mu_d \times (L + R)) + SS$$
4. **Recommended Order Quantity**:
   $$Q = \max(0, S_{target} - \text{Current Inventory})$$

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.9+ (tested up to Python 3.14)
- Git

### 1. Clone & Navigate
```bash
git clone https://github.com/prajwalkumargrti-collab/demand-forecasting-inventory-optimization.git
cd demand-forecasting-inventory-optimization
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
> The dashboard will automatically launch in your browser at `http://localhost:8501`.

---

## 🔌 Decision-Support REST API (FastAPI)

For headless or integrated ERP workflows, a production FastAPI service is included.

### Start the API Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: **`http://localhost:8000/docs`**

### Available Endpoints:
- `GET /health` — Service health and dataset summary.
- `GET /forecast?store_id=S001&product_id=P0001&horizon=30` — Validated demand forecast for specified store and product.
- `GET /inventory?store_id=S001` — Dynamic replenishment parameters, stockout risk scores, and reorder quantities.
- `GET /stores` — List all valid store identifiers.
- `GET /products` — List all valid product identifiers.

#### Example API Request:
```bash
curl -X GET "http://localhost:8000/forecast?store_id=S001&product_id=P0001&horizon=14"
```

---

## 📈 Model Performance Summary

- **Dataset**: 76,000 daily observations across 5 stores and 20 products.
- **Evaluation Split**: Chronological hold-out (last 30 days used as test set to prevent future data leakage).
- **Random Forest Regressor**:
  - **MAE**: ~22.86 units
  - **RMSE**: ~30.82 units
- **Naive Baseline (Last Known Value)**:
  - **MAE**: ~48.61 units
- **Performance Improvement**: **~53% lower forecast error** relative to naive baseline.
