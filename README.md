# Demand Forecasting and Inventory Optimization Using Machine Learning

## Project Overview
This project focuses on preparing retail sales data for demand forecasting and inventory optimization. The goal is to organize and validate historical data so the team can later develop forecasting models and support better inventory decisions.

## Objectives
- Collect and inspect the available retail dataset.
- Identify missing values, duplicate records, invalid dates, and potential data-quality issues.
- Clean and organize the dataset without changing the original raw file.
- Document the dataset fields and data-quality findings.
- Prepare a reliable foundation for the team's next project stages.

## Dataset Summary
- Records: 76,000
- Columns: 16
- Stores: 5
- Products: 20
- Date range: January 1, 2022 – January 30, 2024

The dataset includes fields such as date, store ID, product ID, units sold, inventory level, units ordered, price, promotion, and demand.

*Note: Dataset source, licence, and the exact meanings of certain fields should be verified before further use.*

## Technologies Used
- Python
- Pandas
- Git and GitHub
- Visual Studio Code

## Repository Structure
```text
demand-forecasting-inventory-optimization/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── reports/
│   ├── data_dictionary.md
│   └── data_quality_report.md
├── src/
│   ├── data_preprocessing.py
│   └── prepare_model_data.py
├── .gitignore
└── README.md
```

## Current Progress
- Initial dataset quality checks completed.
- Missing values and exact duplicate rows checked.
- Date values validated.
- Data dictionary prepared.
- Data-quality report prepared.
- Raw and processed CSV files excluded from Git until sharing permissions are confirmed.

## Team Contributions
The project is being developed collaboratively. Tasks are divided among team members covering data preparation, exploratory data analysis, forecasting, inventory optimization, and project integration.

## Important Note
The recorded `Units Sold` and `Demand` values are not assumed to be interchangeable. Their meanings must be validated before model development.

## How to Run the Project

### 1. Clone the repository

git clone https://github.com/prajwalkumargti-collab/demand-forecasting-inventory-optimization.git
cd demand-forecasting-inventory-optimization


### 2. Install dependencies

python -m pip install -r requirements.txt

### 3. Run data preprocessing

python src/data_preprocessing.py

### 4. Check the output
After the script runs successfully, the cleaned dataset will be available at:

`data/processed/cleaned_sales.csv`

### Dataset
The original dataset is available at `data/raw/sales_data.csv`. Keep this file in place before running the preprocessing script.