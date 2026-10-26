
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_sales.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_ready_sales.csv"
)



print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)



required_columns = [
    "Date",
    "Store ID",
    "Product ID",
    "Units Sold",
    "Units Ordered",
    "Demand",
    "Inventory Level",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )



df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

numeric_columns = [
    "Units Sold",
    "Units Ordered",
    "Demand",
    "Inventory Level",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )


df = df.dropna(
    subset=[
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
        "Demand",
    ]
).copy()



duplicate_count = df.duplicated(
    subset=["Date", "Store ID", "Product ID"]
).sum()

print("Duplicate date-store-product records:", duplicate_count)



df = df.sort_values("Date")

model_df = (
    df.groupby(
        ["Date", "Store ID", "Product ID"],
        as_index=False
    )
    .agg(
        Units_Sold=("Units Sold", "sum"),
        Units_Ordered=("Units Ordered", "sum"),
        Demand=("Demand", "sum"),
        Inventory_Level=("Inventory Level", "last"),
    )
)



model_df["Day"] = model_df["Date"].dt.day
model_df["Month"] = model_df["Date"].dt.month
model_df["Year"] = model_df["Date"].dt.year
model_df["DayOfWeek"] = model_df["Date"].dt.dayofweek
model_df["IsWeekend"] = (
    model_df["DayOfWeek"] >= 5
).astype(int)



model_df = model_df.sort_values(
    ["Store ID", "Product ID", "Date"]
).reset_index(drop=True)



print("\nFINAL DATASET VALIDATION")

print("Missing values:")
print(model_df.isnull().sum())

print(
    "\nDuplicate records:",
    model_df.duplicated(
        subset=["Date", "Store ID", "Product ID"]
    ).sum()
)

print("\nDate range:")
print("Start:", model_df["Date"].min())
print("End:", model_df["Date"].max())

print("\nDataset shape:", model_df.shape)



OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

model_df.to_csv(OUTPUT_PATH, index=False)

print("\nModel-ready dataset saved successfully!")
print("Output file:", OUTPUT_PATH)

print("\nFirst five rows:")
print(model_df.head().to_string(index=False))

print("\nPREPARATION COMPLETE")
