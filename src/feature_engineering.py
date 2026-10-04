"""
Feature Engineering Pipeline
----------------------------
Creates domain-specific features for customer churn prediction.
"""

import os
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CLEANED_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cleaned_churn_data.csv"
)

FEATURED_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features_churn_data.csv"
)


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------

def engineer_features(
    df=None,
    save_output=True
):
    """
    Create domain-specific features.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Cleaned dataframe. If None, the cleaned CSV is loaded.

    save_output : bool
        If True, save the engineered dataset to disk.
        Training uses True.
        Prediction uses False.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing engineered features.
    """

    # ---------------------------------------------------------------
    # Load data if no dataframe was provided
    # ---------------------------------------------------------------

    if df is None:
        df = pd.read_csv(
            CLEANED_PATH
        )

    # Always work on a copy.
    df = df.copy()

    # ---------------------------------------------------------------
    # 1. Customer tenure group
    # ---------------------------------------------------------------

    df["tenure_cohort"] = pd.cut(
        df["tenure"],
        bins=[
            -1,
            12,
            24,
            48,
            73
        ],
        labels=[
            "0-12m",
            "13-24m",
            "25-48m",
            "49-72m"
        ]
    )

    # ---------------------------------------------------------------
    # 2. Number of security/support services
    # ---------------------------------------------------------------

    security_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport"
    ]

    df["num_security_services"] = sum(
        (df[col] == "Yes").astype(int)
        for col in security_columns
    )

    # ---------------------------------------------------------------
    # 3. Number of streaming services
    # ---------------------------------------------------------------

    streaming_columns = [
        "StreamingTV",
        "StreamingMovies"
    ]

    df["num_streaming_services"] = sum(
        (df[col] == "Yes").astype(int)
        for col in streaming_columns
    )

    # ---------------------------------------------------------------
    # 4. Total active optional services
    # ---------------------------------------------------------------

    df["total_active_services"] = (
        df["num_security_services"]
        + df["num_streaming_services"]
    )

    # ---------------------------------------------------------------
    # 5. Automatic payment indicator
    # ---------------------------------------------------------------

    df["is_automatic_payment"] = (
        df["PaymentMethod"]
        .astype(str)
        .str.lower()
        .str.contains("automatic")
        .astype(int)
    )

    # ---------------------------------------------------------------
    # 6. Total number of active optimal services
    # ---------------------------------------------------------------

    total_services = (
        (df["PhoneService"] == "Yes").astype(int)
        + (df["InternetService"] != "No").astype(int)
        + df["total_active_services"]
    )

    # ---------------------------------------------------------------
    # 7. Monthly charges per active service
    # ---------------------------------------------------------------

    df["charges_per_active_service"] = (
        df["MonthlyCharges"]
        / total_services.clip(lower=1)
    )

    # ---------------------------------------------------------------
    # 8. Monthly-to-total charge ratio
    # ---------------------------------------------------------------

    df["monthly_to_total_ratio"] = (
        df["MonthlyCharges"]
        / (df["TotalCharges"] + 1.0)
    )

    # ---------------------------------------------------------------
    # 9. Senior citizen living alone
    # ---------------------------------------------------------------

    df["is_senior_living_alone"] = (
        (df["SeniorCitizen"] == 1)
        & (df["Partner"] == "No")
        & (df["Dependents"] == "No")
    ).astype(int)

    # ---------------------------------------------------------------
    # Save feature dataset only when requested
    # ---------------------------------------------------------------

    if save_output:

        os.makedirs(
            os.path.dirname(FEATURED_PATH),
            exist_ok=True
        )

        df.to_csv(
            FEATURED_PATH,
            index=False
        )

        print(
            f"[INFO] Feature dataset saved to: "
            f"{FEATURED_PATH}"
        )

    print(
        f"[INFO] Feature dataset shape: {df.shape}"
    )

    return df


# -------------------------------------------------------------------
# Run directly
# -------------------------------------------------------------------

if __name__ == "__main__":

    from data_preprocessing import clean_data

    cleaned_df = clean_data()

    engineer_features(
        cleaned_df,
        save_output=True
    )