"""
Configuration constants for the Demand Forecasting & Inventory Optimization Dashboard.
"""

import os
from pathlib import Path

# ── Project Root & Data Path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_REPO_DATA = PROJECT_ROOT / "data" / "raw" / "sales_data.csv"
_LOCAL_DOWNLOAD = Path(r"C:\Users\samal\Downloads\sales_data.csv")

if _REPO_DATA.exists():
    DATA_PATH = str(_REPO_DATA)
elif _LOCAL_DOWNLOAD.exists():
    DATA_PATH = str(_LOCAL_DOWNLOAD)
else:
    DATA_PATH = str(_REPO_DATA)

# ── Inventory optimisation parameters ────────────────────────────────────────
LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL_Z = 1.65          # ~95 % service level
LOOKBACK_DAYS = 28              # rolling window for demand estimation

# ── Forecasting defaults ─────────────────────────────────────────────────────
DEFAULT_FORECAST_HORIZON = 30   # days ahead

# ── Dashboard colour palette (professional blue-teal-grey) ───────────────────
COLOR_PALETTE = {
    "primary":        "#1B2A4A",
    "secondary":      "#2E86AB",
    "accent":         "#F18F01",
    "success":        "#28A745",
    "warning":        "#FFC107",
    "danger":         "#DC3545",
    "light_bg":       "#F8F9FA",
    "dark_text":      "#212529",
    "muted_text":     "#6C757D",
    "chart_colors": [
        "#2E86AB",  # teal-blue
        "#F18F01",  # orange
        "#A23B72",  # plum
        "#28A745",  # green
        "#DC3545",  # red
        "#6F42C1",  # purple
        "#17A2B8",  # cyan
        "#FD7E14",  # deep-orange
        "#20C997",  # mint
        "#E83E8C",  # pink
    ],
}

# ── Stock status thresholds ──────────────────────────────────────────────────
STOCK_STATUS_REORDER = "REORDER NOW"
STOCK_STATUS_LOW     = "LOW STOCK"
STOCK_STATUS_OK      = "SUFFICIENT"
