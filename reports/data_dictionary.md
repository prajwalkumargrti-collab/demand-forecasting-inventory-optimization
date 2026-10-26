
# Data Dictionary

## Project
Demand Forecasting and Inventory Optimization Using Machine Learning

## Dataset
Source file: `data/raw/sales_data.csv`

Total records: 76,000

Total columns: 16

## Column Definitions

| Column | Description | Data Type |
|---|---|---|
| Date | Date associated with the sales record | Datetime |
| Store ID | Identifier for the store | String |
| Product ID | Identifier for the product | String |
| Category | Product category | String |
| Region | Geographic region associated with the store | String |
| Inventory Level | Recorded inventory level | Integer |
| Units Sold | Number of units sold | Integer |
| Units Ordered | Number of units ordered | Integer |
| Price | Recorded product price | Float |
| Discount | Recorded discount value | Integer |
| Weather Condition | Weather information associated with the record | String |
| Promotion | Promotion indicator or value | Integer |
| Competitor Pricing | Recorded competitor price | Float |
| Seasonality | Seasonal information associated with the record | String |
| Epidemic | Epidemic-related indicator or value | Integer |
| Demand | Demand value recorded in the dataset | Integer |

## Notes

- The definitions above describe the apparent role of each field; confirm exact meanings and units against the dataset documentation.
- `Units Sold` and `Demand` must not be treated as interchangeable without validation.
- The dataset contains 760 unique dates, 5 stores and 20 products.
- No missing values or exact duplicate rows were found in the initial checks.
- The original CSV is preserved separately from the cleaned output.
