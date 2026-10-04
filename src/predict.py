"""
Inference Module
----------------
Loads the trained churn prediction pipeline and performs
predictions on new customer data.
"""

import os
import json
import joblib
import pandas as pd

from feature_engineering import engineer_features


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_churn_model.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_metrics.json"
)


class ChurnPredictor:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

        with open(
            METRICS_PATH,
            "r"
        ) as f:

            self.metrics = json.load(f)

        self.threshold = self.metrics.get(
            "best_threshold",
            0.5
        )

    def predict_customer(
        self,
        data: dict,
        threshold=None
    ):
        """
        Predict churn probability for one customer.
        """

        if threshold is None:
            threshold = self.threshold

        # Convert input dictionary into dataframe.
        df = pd.DataFrame([data])

        # Apply the same feature engineering used during training.
        df = engineer_features(
            df,
            save_output=False
        )

        # Remove target if accidentally provided.
        if "Churn" in df.columns:
            df = df.drop(
                columns=["Churn"]
            )

        # Predict churn probability.
        probability = float(
            self.model.predict_proba(df)[0, 1]
        )

        prediction = int(
            probability >= threshold
        )

        # Risk category.
        if probability < 0.35:
            risk_tier = "Low Risk"

        elif probability < 0.65:
            risk_tier = "Medium Risk"

        else:
            risk_tier = "High Risk"

        # Business-oriented risk indicators.
        # These are rule-based indicators and are separate
        # from the model's probability calculation.
        drivers = []

        if data.get("Contract") == "Month-to-month":
            drivers.append(
                "Month-to-month contract"
            )

        if float(
            data.get("tenure", 0)
        ) <= 12:

            drivers.append(
                "Short customer tenure"
            )

        if data.get(
            "InternetService"
        ) == "Fiber optic":

            drivers.append(
                "Fiber optic internet service"
            )

        if data.get(
            "PaymentMethod"
        ) == "Electronic check":

            drivers.append(
                "Electronic check payment"
            )

        if data.get(
            "OnlineSecurity"
        ) == "No":

            drivers.append(
                "No online security service"
            )

        if data.get(
            "TechSupport"
        ) == "No":

            drivers.append(
                "No technical support service"
            )

        if not drivers:
            drivers.append(
                "No major rule-based risk indicators"
            )

        # Retention recommendation.
        if risk_tier == "High Risk":

            recommendation = (
                "Prioritize immediate retention outreach "
                "with a contract or pricing incentive."
            )

        elif risk_tier == "Medium Risk":

            recommendation = (
                "Offer a service bundle or loyalty "
                "incentive and monitor the customer."
            )

        else:

            recommendation = (
                "Continue standard loyalty and "
                "customer engagement activities."
            )

        return {
            "churn_probability":
                round(probability, 4),

            "churn_prediction":
                bool(prediction),

            "risk_tier":
                risk_tier,

            "threshold":
                threshold,

            "risk_indicators":
                drivers,

            "retention_recommendation":
                recommendation
        }


if __name__ == "__main__":

    sample = {

        "gender": "Female",

        "SeniorCitizen": 0,

        "Partner": "No",

        "Dependents": "No",

        "tenure": 2,

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": "Fiber optic",

        "OnlineSecurity": "No",

        "OnlineBackup": "No",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "Yes",

        "StreamingMovies": "Yes",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod":
            "Electronic check",

        "MonthlyCharges": 92.50,

        "TotalCharges": 185.00
    }

    predictor = ChurnPredictor()

    result = predictor.predict_customer(
        sample
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )