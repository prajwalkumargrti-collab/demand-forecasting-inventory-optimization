from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sales_data.csv"


df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())
print("\nData information:")
df.info()
print("\nDuplicate rows:", df.duplicated().sum())
print("\nNegative values check:")

numeric_columns = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Competitor Pricing",
    "Demand"
]

for column in numeric_columns:
    if column in df.columns:
        negative_count = (df[column] < 0).sum()
        print(f"{column}: {negative_count}")
        print("\nDate validation:")

dates = pd.to_datetime(df["Date"], errors="coerce")

print("Invalid dates:", dates.isna().sum())
print("Earliest date:", dates.min())
print("Latest date:", dates.max())
print("\nDate coverage check:")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("Unique dates:", df["Date"].nunique())
print("Unique stores:", df["Store ID"].nunique())
print("Unique products:", df["Product ID"].nunique())

print(
    "Unique store-product combinations:",
    df[["Store ID", "Product ID"]].drop_duplicates().shape[0]
)
print("\nMissing daily records check:")

expected_rows = (
    df["Date"].nunique()
    * df[["Store ID", "Product ID"]]
        .drop_duplicates()
        .shape[0]
)

actual_rows = df[
    ["Date", "Store ID", "Product ID"]
].drop_duplicates().shape[0]

print("Expected daily records:", expected_rows)
print("Actual unique daily records:", actual_rows)
print("Missing daily records:", expected_rows - actual_rows)
print("\nFinal missing-value check:")

missing_values = df.isnull().sum()

print(missing_values)
print("Total missing values:", missing_values.sum())
print("\nData type check:")
print(df.dtypes)

print("\nDataset dimensions:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nStarting data cleaning...")


cleaned_df = df.copy()

cleaned_df = cleaned_df.drop_duplicates()

cleaned_df["Date"] = pd.to_datetime(
    cleaned_df["Date"], errors="coerce"
)
cleaned_df = cleaned_df.dropna(subset=["Date"])

cleaned_df = cleaned_df.sort_values(
    by=["Date", "Store ID", "Product ID"]
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_sales.csv"

cleaned_df.to_csv(OUTPUT_PATH, index=False)

print("Data cleaning completed!")
print("Original rows:", len(df))
print("Cleaned rows:", len(cleaned_df))
print("Saved to:", OUTPUT_PATH)
# Final data quality validation
print("\nFINAL DATA QUALITY REPORT")

print("Missing values:", cleaned_df.isnull().sum().sum())
print("Duplicate rows:", cleaned_df.duplicated().sum())

print("\nNegative values:")
for column in [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Competitor Pricing",
    "Demand"
]:
    if column in cleaned_df.columns:
        print(f"{column}: {(cleaned_df[column] < 0).sum()}")

print("\nDate range:")
print("Start:", cleaned_df["Date"].min())
print("End:", cleaned_df["Date"].max())

print("\nFinal dataset shape:", cleaned_df.shape)
print("\nSALES AND DEMAND VALIDATION")

print("Units Sold summary:")
print(cleaned_df["Units Sold"].describe())

print("\nDemand summary:")
print(cleaned_df["Demand"].describe())

print("\nZero Units Sold:",
      (cleaned_df["Units Sold"] == 0).sum())

print("Zero Demand:",
      (cleaned_df["Demand"] == 0).sum())

print("\nUnits Sold greater than Demand:",
      (cleaned_df["Units Sold"] > cleaned_df["Demand"]).sum())

print("Demand greater than Units Sold:",
      (cleaned_df["Demand"] > cleaned_df["Units Sold"]).sum())
print("\nSALES-DEMAND RELATIONSHIP")

print(
    cleaned_df[
        ["Units Sold", "Units Ordered", "Demand", "Inventory Level"]
    ].head(10).to_string(index=False)
)

print("\nCorrelation between Units Sold and Demand:")
print(cleaned_df["Units Sold"].corr(cleaned_df["Demand"]))

print("\nRows where Units Sold exceeds Demand:")
print(
    cleaned_df.loc[
        cleaned_df["Units Sold"] > cleaned_df["Demand"],
        ["Date", "Store ID", "Product ID",
         "Units Sold", "Units Ordered", "Demand"]
    ].head(10).to_string(index=False)
)
print("\nCOLUMN RELATIONSHIP CHECK")

print("Units Sold equals Demand:",
      (cleaned_df["Units Sold"] == cleaned_df["Demand"]).sum())

print("Units Sold equals Units Ordered:",
      (cleaned_df["Units Sold"] == cleaned_df["Units Ordered"]).sum())

print("\nSample records:")
print(
    cleaned_df[
        ["Date", "Store ID", "Product ID",
         "Inventory Level", "Units Sold",
         "Units Ordered", "Demand"]
    ].head(10).to_string(index=False)
)