"""
Data Cleaning Pipeline
----------------------
Loads the raw Telco Customer Churn dataset and performs
basic data cleaning before feature engineering.
"""

import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Telco-Customer-Churn.csv"
)

CLEANED_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_churn_data.csv"
)


def clean_data():
    """
    Load and clean the raw Telco dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe.
    """

    df = pd.read_csv(RAW_PATH)

    # Convert TotalCharges to numeric.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    # Customers with zero tenure have no accumulated charges.
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0

    # Remove the customer identifier.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Any remaining missing TotalCharges values are filled
    # using the median of the cleaned dataset.
    if df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)

    df.to_csv(CLEANED_PATH, index=False)

    print(
        f"[INFO] Cleaned dataset saved to: {CLEANED_PATH}"
    )
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Missing values: {df.isna().sum().sum()}")

    return df


if __name__ == "__main__":
    clean_data()