"""
Telco Customer Churn Prediction
--------------------------------
Streamlit web application for customer churn
risk assessment and retention recommendations.
"""

import os
import sys
import json
import pandas as pd

import streamlit as st


# ============================================================
# Project configuration
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_metrics.json"
)

sys.path.insert(
    0,
    SRC_DIR
)

from predict import ChurnPredictor


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Load model metrics
# ============================================================

if not os.path.exists(METRICS_PATH):

    st.error(
        "Model metrics file not found. "
        "Please run `python src/train.py` first."
    )

    st.stop()


with open(
    METRICS_PATH,
    "r"
) as f:

    metrics = json.load(f)


# ============================================================
# Production model settings
# ============================================================

production_threshold = metrics.get(
    "best_threshold",
    0.56
)

champion_model = metrics.get(
    "best_model_display",
    "Logistic Regression"
)


# ============================================================
# Page header
# ============================================================

st.title(
    "📊 Telco Customer Churn Intelligence"
)

st.markdown(
    """
Predict customer churn risk, understand the main risk drivers,
and generate a practical retention recommendation.
"""
)


# ============================================================
# Sidebar — Customer information
# ============================================================

st.sidebar.header(
    "Customer Profile"
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multilines = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "No",
        "Yes",
        "No phone service"
    ]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    [
        "Fiber optic",
        "DSL",
        "No"
    ]
)

sec = st.sidebar.selectbox(
    "Online Security",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

protect = st.sidebar.selectbox(
    "Device Protection",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

support = st.sidebar.selectbox(
    "Tech Support",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

contract = st.sidebar.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.slider(
    "Monthly Charges ($)",
    18.0,
    120.0,
    75.0,
    0.5
)

total = st.sidebar.number_input(
    "Total Charges ($)",
    min_value=0.0,
    max_value=9000.0,
    value=float(
        tenure * monthly
    )
)


# ============================================================
# Prediction threshold
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Model Configuration"
)

st.sidebar.info(
    f"Production threshold: **{production_threshold:.2f}**"
)

st.sidebar.caption(
    "The threshold was selected during model training "
    "to balance churn detection performance."
)


# ============================================================
# Create customer dictionary
# ============================================================

cust_dict = {

    "gender": gender,

    "SeniorCitizen": senior,

    "Partner": partner,

    "Dependents": dependents,

    "tenure": tenure,

    "PhoneService": phone,

    "MultipleLines": multilines,

    "InternetService": internet,

    "OnlineSecurity": sec,

    "OnlineBackup": backup,

    "DeviceProtection": protect,

    "TechSupport": support,

    "StreamingTV": tv,

    "StreamingMovies": movies,

    "Contract": contract,

    "PaperlessBilling": paperless,

    "PaymentMethod": payment,

    "MonthlyCharges": monthly,

    "TotalCharges": total
}


# ============================================================
# Run prediction
# ============================================================

predictor = ChurnPredictor()

try:

    result = predictor.predict_customer(
        cust_dict,
        threshold=production_threshold
    )

except Exception as e:

    st.error(
        "Prediction failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# Results
# ============================================================

col1, col2 = st.columns(
    [1, 1]
)


# ============================================================
# Customer risk assessment
# ============================================================

with col1:

    st.subheader(
        "🎯 Customer Risk Assessment"
    )

    probability = result[
        "churn_probability"
    ]

    prediction = result[
        "churn_prediction"
    ]

    risk_tier = result[
        "risk_tier"
    ]

    # --------------------------------------------------------
    # Risk display
    # --------------------------------------------------------

    if probability >= 0.65:

        st.error(
            f"🚨 High Churn Risk — "
            f"{probability * 100:.1f}%"
        )

    elif probability >= 0.35:

        st.warning(
            f"⚠️ Moderate Churn Risk — "
            f"{probability * 100:.1f}%"
        )

    else:

        st.success(
            f"✅ Low Churn Risk — "
            f"{probability * 100:.1f}%"
        )

    st.progress(
        float(probability)
    )

    # --------------------------------------------------------
    # Key metrics
    # --------------------------------------------------------

    metric1, metric2 = st.columns(
        2
    )

    with metric1:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

    with metric2:

        st.metric(
            "Decision Threshold",
            f"{production_threshold * 100:.0f}%"
        )
    st.caption(
        "The customer is classified as likely to churn when "
        f"the predicted probability reaches the "
        f"{production_threshold * 100:.0f}% decision threshold."
    )
    st.write(
        f"**Prediction:** "
        f"{'Likely to Churn' if prediction else 'Likely to Stay'}"
    )

    st.write(
        f"**Risk Tier:** {risk_tier}"
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.subheader(
        "💡 Retention Recommendation"
    )

    st.info(
        result[
            "retention_recommendation"
        ]
    )

    # --------------------------------------------------------
    # Business risk indicators
    # --------------------------------------------------------

    st.subheader(
        "🔍 Business Risk Indicators"
    )

    risk_indicators = result.get(
        "risk_indicators",
        []
    )

    if risk_indicators:

        for indicator in risk_indicators:

            st.markdown(
                f"- 🔴 {indicator}"
            )

    else:

        st.write(
            "No major rule-based risk indicator identified."
        )
    st.caption(
        "These indicators are business rules shown alongside "
        "the ML prediction; they are not direct model explations."
    )


# ============================================================
# Model benchmark
# ============================================================

with col2:

    st.subheader(
        "📈 Model Benchmark"
    )

    model_rows = []

    for model_name, model_info in metrics[
        "models"
    ].items():

        model_rows.append(
            {
                "Model": model_info[
                    "display_name"
                ],

                "CV ROC-AUC": model_info.get(
                    "cv_roc_auc",
                    model_info.get(
                        "roc_auc",
                        0
                    )
                ),

                "Test ROC-AUC": model_info.get(
                    "test_roc_auc",
                    model_info.get(
                        "roc_auc",
                        0
                    )
                ),

                "F1-Score": model_info.get(
                    "f1_score",
                    0
                ),

                "Recall": model_info.get(
                    "recall",
                    0
                )
            }
        )

    model_df = pd.DataFrame(
        model_rows
    )

    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"🏆 Champion Model: **{champion_model}**"
    )

    st.caption(
        f"Production decision threshold: "
        f"{production_threshold:.2f}"
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "Telco Customer Churn Prediction | "
    "Machine Learning + Streamlit"
)