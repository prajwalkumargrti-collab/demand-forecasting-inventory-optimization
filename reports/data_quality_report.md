
# Data Quality Report

## 1. Project Information

**Project:** Demand Forecasting and Inventory Optimization Using Machine Learning

**Role:** Data Collection and Data Preprocessing

**Source file:** `data/raw/sales_data.csv`

**Cleaned file:** `data/processed/cleaned_sales.csv`

## 2. Dataset Overview

| Attribute | Result |
|---|---|
| Total records | 76,000 |
| Total columns | 16 |
| Unique stores | 5 |
| Unique products | 20 |
| Unique dates | 760 |
| Start date | 2022-01-01 |
| End date | 2024-01-30 |

## 3. Data Quality Checks

| Check | Result | Status |
|---|---:|---|
| Missing values | 0 | Passed |
| Exact duplicate rows | 0 | Passed |
| Invalid dates | 0 | Passed |
| Negative values in checked numeric columns | 0 | Passed |
| Missing daily records under the full-coverage assumption | 0 | Passed |
| Rows after cleaning | 76,000 | Verified |

## 4. Columns Reviewed

The dataset contains the following columns:

- Date
- Store ID
- Product ID
- Category
- Region
- Inventory Level
- Units Sold
- Units Ordered
- Price
- Discount
- Weather Condition
- Promotion
- Competitor Pricing
- Seasonality
- Epidemic
- Demand

## 5. Preprocessing Performed

1. Loaded the original CSV without modifying the source file.
2. Checked column names, data types and missing values.
3. Checked for exact duplicate rows.
4. Validated date values and the date range.
5. Checked negative values in selected numerical columns.
6. Checked store-product combinations and daily record coverage.
7. Converted the Date column to a datetime type.
8. Sorted the cleaned data by date, store and product.
9. Saved the cleaned dataset separately.

## 6. Output Files

- `data/raw/sales_data.csv` — original dataset.
- `data/processed/cleaned_sales.csv` — cleaned dataset.
- `src/data_preprocessing.py` — preprocessing script.

## 7. Limitations and Pending Checks

- The meaning and relationship between `Units Sold` and `Demand` require confirmation before model training.
- The absence of negative values does not prove that every value is logically correct.
- Daily record completeness was assessed assuming that every store-product combination should appear on every date.
- Supplier lead times, service-level assumptions and the interpretation of inventory fields must be confirmed before calculating replenishment recommendations.
- The dataset's source and whether it is real or synthetic should be documented and verified.

## 8. Conclusion

The dataset passed the initial checks for missing values, exact duplicate rows, invalid dates and negative values in the selected numerical columns. The cleaned dataset was saved separately from the original source.

Further checks are required to confirm business meanings and forecasting assumptions before the dataset is used for model training.
