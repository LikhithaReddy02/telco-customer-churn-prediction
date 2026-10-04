```python
"""
Data Acquisition Script
-----------------------
Downloads the IBM Telco Customer Churn dataset if it is not
already available locally.
"""

import os
import urllib.request


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

RAW_PATH = os.path.join(
    RAW_DIR,
    "Telco-Customer-Churn.csv"
)


# -------------------------------------------------------------------
# Dataset source
# -------------------------------------------------------------------

PRIMARY_URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


# -------------------------------------------------------------------
# Download function
# -------------------------------------------------------------------

def download_data():
    """
    Download the raw Telco Customer Churn dataset.

    Returns
    -------
    bool
        True if the dataset exists/download succeeds,
        otherwise False.
    """

    # Create raw data directory if necessary.
    os.makedirs(
        RAW_DIR,
        exist_ok=True
    )

    # Avoid downloading the file again if it already exists.
    if os.path.exists(RAW_PATH):

        print(
            f"[INFO] Raw dataset already exists at: "
            f"{RAW_PATH}"
        )

        return True

    try:

        print(
            f"[INFO] Downloading dataset from:\n"
            f"{PRIMARY_URL}"
        )

        urllib.request.urlretrieve(
            PRIMARY_URL,
            RAW_PATH
        )

        print(
            "[SUCCESS] Dataset downloaded successfully."
        )

        print(
            f"[INFO] Saved to: {RAW_PATH}"
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] Dataset download failed: {e}"
        )

        # Remove incomplete file if one was created.
        if os.path.exists(RAW_PATH):
            os.remove(RAW_PATH)

        return False


# -------------------------------------------------------------------
# Run directly
# -------------------------------------------------------------------

if __name__ == "__main__":
    download_data()
```
