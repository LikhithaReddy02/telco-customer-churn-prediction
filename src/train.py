"""
Model Training Pipeline
-----------------------
Telco Customer Churn Prediction

Pipeline:

Raw Data
    ↓
Cleaning
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
ColumnTransformer
    ↓
Model Benchmarking
    ↓
Stratified 5-Fold Cross Validation
    ↓
Model Selection
    ↓
Threshold Tuning
    ↓
Final Test Evaluation
    ↓
Model Serialization
"""

import os
import sys
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# Allow imports from src/
sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from data_preprocessing import clean_data
from feature_engineering import engineer_features
from models import get_model


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_churn_model.pkl"
)

PREP_PATH = os.path.join(
    MODELS_DIR,
    "preprocessor.pkl"
)

METRICS_PATH = os.path.join(
    MODELS_DIR,
    "model_metrics.json"
)


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def evaluate_predictions(y_true, probabilities, threshold=0.5):
    """
    Calculate classification metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(
            accuracy_score(y_true, predictions), 4
        ),
        "precision": round(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),
            4
        ),
        "recall": round(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),
            4
        ),
        "f1_score": round(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),
            4
        ),
        "roc_auc": round(
            roc_auc_score(
                y_true,
                probabilities
            ),
            4
        ),
        "cm": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }


# -------------------------------------------------------------------
# Threshold tuning
# -------------------------------------------------------------------

def find_best_threshold(y_true, probabilities):
    """
    Find the threshold that produces the highest F1 score.
    """

    thresholds = np.arange(
        0.20,
        0.81,
        0.01
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return round(
        float(best_threshold),
        2
    )


# -------------------------------------------------------------------
# Main training
# -------------------------------------------------------------------

def main():

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    print("\n" + "=" * 80)
    print("TELCO CUSTOMER CHURN - MODEL TRAINING")
    print("=" * 80)

    # ---------------------------------------------------------------
    # 1. Data cleaning
    # ---------------------------------------------------------------

    print("\n[1/7] Cleaning data...")

    df_clean = clean_data()

    # ---------------------------------------------------------------
    # 2. Feature engineering
    # ---------------------------------------------------------------

    print("\n[2/7] Engineering features...")

    df = engineer_features(
        df_clean,
        save_output=True
    )

    # ---------------------------------------------------------------
    # 3. Prepare target
    # ---------------------------------------------------------------

    print("\n[3/7] Preparing target variable...")

    y = (
        df["Churn"] == "Yes"
    ).astype(int)

    X = df.drop(
        columns=["Churn"]
    )

    print(f"[INFO] Features: {X.shape[1]}")
    print(f"[INFO] Samples: {X.shape[0]}")
    print(
        f"[INFO] Churn rate: {y.mean():.2%}"
    )

    # ---------------------------------------------------------------
    # 4. Train/test split
    # ---------------------------------------------------------------

    print("\n[4/7] Creating stratified train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print(
        f"[INFO] Training samples: {len(X_train)}"
    )

    print(
        f"[INFO] Test samples: {len(X_test)}"
    )

    # ---------------------------------------------------------------
    # 5. Preprocessing
    # ---------------------------------------------------------------

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    print(
        f"[INFO] Numerical features: "
        f"{len(numeric_features)}"
    )

    print(
        f"[INFO] Categorical features: "
        f"{len(categorical_features)}"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ],
        remainder="drop"
    )

    # ---------------------------------------------------------------
    # 6. Model benchmarking
    # ---------------------------------------------------------------

    print("\n[5/7] Running model benchmarking...")
    print("-" * 80)

    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting"
    ]

    display_names = {
        "logistic_regression":
            "Logistic Regression",

        "decision_tree":
            "Decision Tree",

        "random_forest":
            "Random Forest",

        "gradient_boosting":
            "Gradient Boosting"
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    metrics_summary = {}

    best_model_name = None
    best_cv_auc = -1.0

    for model_name in model_names:

        print(
            f"\n[MODEL] {display_names[model_name]}"
        )

        model = get_model(
            model_name,
            random_state=42
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        start_time = time.time()

        # -----------------------------------------------------------
        # Real 5-fold cross-validation
        # -----------------------------------------------------------

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "f1": "f1",
                "recall": "recall",
                "precision": "precision"
            },
            n_jobs=-1
        )

        cv_auc = float(
            np.mean(
                cv_results["test_roc_auc"]
            )
        )

        cv_f1 = float(
            np.mean(
                cv_results["test_f1"]
            )
        )

        cv_recall = float(
            np.mean(
                cv_results["test_recall"]
            )
        )

        cv_precision = float(
            np.mean(
                cv_results["test_precision"]
            )
        )

        # -----------------------------------------------------------
        # Out-of-fold probabilities for threshold tuning
        # -----------------------------------------------------------

        oof_probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        best_threshold = find_best_threshold(
            y_train,
            oof_probabilities
        )

        # -----------------------------------------------------------
        # Fit on complete training data
        # -----------------------------------------------------------

        pipeline.fit(
            X_train,
            y_train
        )

        test_probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        test_metrics = evaluate_predictions(
            y_test,
            test_probabilities,
            threshold=best_threshold
        )

        training_time = round(
            time.time() - start_time,
            3
        )

        metrics_summary[model_name] = {
            "display_name":
                display_names[model_name],

            "cv_roc_auc":
                round(cv_auc, 4),

            "cv_f1":
                round(cv_f1, 4),

            "cv_recall":
                round(cv_recall, 4),

            "cv_precision":
                round(cv_precision, 4),

            "best_threshold":
                best_threshold,

            "accuracy":
                test_metrics["accuracy"],

            "precision":
                test_metrics["precision"],

            "recall":
                test_metrics["recall"],

            "f1_score":
                test_metrics["f1_score"],

            "roc_auc":
                test_metrics["roc_auc"],

            "cm":
                test_metrics["cm"],

            "train_time_sec":
                training_time
        }

        print(
            f"CV ROC-AUC : {cv_auc:.4f}"
        )

        print(
            f"CV F1      : {cv_f1:.4f}"
        )

        print(
            f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}"
        )

        print(
            f"Test F1     : {test_metrics['f1_score']:.4f}"
        )

        print(
            f"Test Recall : {test_metrics['recall']:.4f}"
        )

        print(
            f"Threshold   : {best_threshold:.2f}"
        )

        # Select model using CV ROC-AUC.
        if cv_auc > best_cv_auc:
            best_cv_auc = cv_auc
            best_model_name = model_name

    # ---------------------------------------------------------------
    # 7. Train final champion model
    # ---------------------------------------------------------------

    print("\n[6/7] Training final champion model...")

    best_model = get_model(
        best_model_name,
        random_state=42
    )

    final_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                best_model
            )
        ]
    )

    final_pipeline.fit(
        X_train,
        y_train
    )

    # Save complete pipeline.
    joblib.dump(
        final_pipeline,
        BEST_MODEL_PATH
    )

    # Save preprocessing object separately for inspection.
    joblib.dump(
        final_pipeline.named_steps["preprocessor"],
        PREP_PATH
    )

    # ---------------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------------

    best_threshold = metrics_summary[
        best_model_name
    ]["best_threshold"]

    full_metrics = {
        "best_model":
            best_model_name,

        "best_model_display":
            display_names[best_model_name],

        "selection_metric":
            "5-fold cross-validation ROC-AUC",

        "best_cv_roc_auc":
            round(best_cv_auc, 4),

        "models":
            metrics_summary,

        "test_size":
            0.20,

        "random_state":
            42,

        "cv_folds":
            5,

        "best_threshold":
            best_threshold,

        "feature_names":
            X.columns.tolist()
    }

    with open(
        METRICS_PATH,
        "w"
    ) as f:

        json.dump(
            full_metrics,
            f,
            indent=2
        )

    print("\n[7/7] Training complete.")
    print("=" * 80)

    print(
        f"Champion Model: "
        f"{display_names[best_model_name]}"
    )

    print(
        f"CV ROC-AUC: "
        f"{best_cv_auc:.4f}"
    )

    print(
        f"Decision Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Model saved to: "
        f"{BEST_MODEL_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()