"""
Data loading and preprocessing utilities for the sales-demand dataset.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd


# ── Loading ──────────────────────────────────────────────────────────────────

def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw sales CSV and parse the Date column.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with Date parsed as datetime.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe and add calendar / time features.

    Steps
    -----
    1. Drop exact duplicate rows.
    2. Ensure ``Date`` is datetime dtype.
    3. Sort by Date → Store ID → Product ID.
    4. Add derived time features:
       Year, Month, DayOfWeek (0=Mon), WeekOfYear, IsWeekend, Quarter.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe returned by :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with extra time-feature columns.
    """
    df = df.drop_duplicates().copy()

    # Guarantee datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True)

    # Calendar features
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["DayOfWeek"]  = df["Date"].dt.dayofweek          # 0 = Monday
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"]    = df["Date"].dt.quarter

    return df


# ── Filtering ────────────────────────────────────────────────────────────────

def get_filtered_data(
    df: pd.DataFrame,
    stores: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    date_range: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """Return a filtered copy of *df* based on any combination of criteria.

    Parameters
    ----------
    df : pd.DataFrame
        The (preprocessed) sales dataframe.
    stores : list[str], optional
        Store IDs to keep (e.g. ``["S001", "S003"]``).
    products : list[str], optional
        Product IDs to keep.
    categories : list[str], optional
        Category names to keep.
    date_range : tuple[str, str], optional
        ``(start_date, end_date)`` strings parsable by ``pd.to_datetime``.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe (copy).
    """
    filtered = df.copy()

    if stores is not None and len(stores) > 0:
        filtered = filtered[filtered["Store ID"].isin(stores)]

    if products is not None and len(products) > 0:
        filtered = filtered[filtered["Product ID"].isin(products)]

    if categories is not None and len(categories) > 0:
        filtered = filtered[filtered["Category"].isin(categories)]

    if date_range is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["Date"] >= start) & (filtered["Date"] <= end)]

    return filtered.reset_index(drop=True)
