"""
Evaluation and Diagnostic Artifact Generation
----------------------------------------------
Generates evaluation summaries and diagnostic plots
for the trained Telco Customer Churn model.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from data_preprocessing import clean_data
from feature_engineering import engineer_features


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FIG_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "figures"
)

METRICS_PATH = os.path.join(
    MODELS_DIR,
    "model_metrics.json"
)

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_churn_model.pkl"
)


# -------------------------------------------------------------------
# Create output directory
# -------------------------------------------------------------------

os.makedirs(
    FIG_DIR,
    exist_ok=True
)


# -------------------------------------------------------------------
# Load metrics
# -------------------------------------------------------------------

def load_metrics():
    """
    Load model evaluation metrics.
    """

    with open(
        METRICS_PATH,
        "r"
    ) as f:

        return json.load(f)


# -------------------------------------------------------------------
# Print model benchmark
# -------------------------------------------------------------------

def print_model_benchmark(metrics_data):
    """
    Print the benchmark results stored during training.
    """

    print("\n" + "=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)

    for model_name, model_info in metrics_data["models"].items():

        print(
            f"\n{model_info['display_name']}"
        )

        print(
            f"  CV ROC-AUC : "
            f"{model_info['cv_roc_auc']:.4f}"
        )

        print(
            f"  Test ROC-AUC: "
            f"{model_info['roc_auc']:.4f}"
        )

        print(
            f"  Test F1     : "
            f"{model_info['f1_score']:.4f}"
        )

        print(
            f"  Test Recall : "
            f"{model_info['recall']:.4f}"
        )

        print(
            f"  Threshold   : "
            f"{model_info['best_threshold']:.2f}"
        )

    print("\n" + "-" * 80)

    print(
        f"Champion Model: "
        f"{metrics_data['best_model_display']}"
    )

    print(
        f"Champion CV ROC-AUC: "
        f"{metrics_data['best_cv_roc_auc']:.4f}"
    )

    print(
        f"Production Threshold: "
        f"{metrics_data['best_threshold']:.2f}"
    )

    print("=" * 80)


# -------------------------------------------------------------------
# Generate confusion matrix
# -------------------------------------------------------------------

def generate_confusion_matrix(metrics_data):
    """
    Generate a confusion matrix from the stored test metrics.
    """

    champion_name = metrics_data["best_model"]

    champion_metrics = metrics_data[
        "models"
    ][champion_name]

    cm_data = champion_metrics["cm"]

    cm = np.array([
        [
            cm_data["TN"],
            cm_data["FP"]
        ],
        [
            cm_data["FN"],
            cm_data["TP"]
        ]
    ])

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Stayed",
            "Churned"
        ]
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        "Champion Model Confusion Matrix"
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIG_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[INFO] Saved: {output_path}"
    )


# -------------------------------------------------------------------
# Generate model comparison chart
# -------------------------------------------------------------------

def generate_model_comparison(metrics_data):
    """
    Generate a model benchmark comparison chart.
    """

    rows = []

    for model_info in metrics_data[
        "models"
    ].values():

        rows.append(
            {
                "Model":
                    model_info["display_name"],

                "CV ROC-AUC":
                    model_info["cv_roc_auc"],

                "Test ROC-AUC":
                    model_info["roc_auc"],

                "F1-Score":
                    model_info["f1_score"],

                "Recall":
                    model_info["recall"]
            }
        )

    df = pd.DataFrame(rows)

    ax = df.set_index(
        "Model"
    ).plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Model Performance Comparison"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIG_DIR,
        "model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[INFO] Saved: {output_path}"
    )


# -------------------------------------------------------------------
# Generate churn distribution
# -------------------------------------------------------------------

def generate_churn_distribution():
    """
    Generate a chart showing the target class distribution.
    """

    df = clean_data()

    churn_counts = df[
        "Churn"
    ].value_counts()

    plt.figure(
        figsize=(7, 5)
    )

    sns.barplot(
        x=churn_counts.index,
        y=churn_counts.values
    )

    plt.title(
        "Customer Churn Distribution"
    )

    plt.xlabel(
        "Churn"
    )

    plt.ylabel(
        "Number of Customers"
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIG_DIR,
        "churn_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[INFO] Saved: {output_path}"
    )


# -------------------------------------------------------------------
# Generate tenure distribution
# -------------------------------------------------------------------

def generate_tenure_distribution():
    """
    Generate a tenure distribution by churn status.
    """

    df = clean_data()

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        bins=30,
        kde=True
    )

    plt.title(
        "Customer Tenure Distribution by Churn Status"
    )

    plt.xlabel(
        "Tenure (Months)"
    )

    plt.ylabel(
        "Number of Customers"
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIG_DIR,
        "tenure_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[INFO] Saved: {output_path}"
    )


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def evaluate_models():

    print(
        "\n[INFO] Loading model metrics..."
    )

    metrics_data = load_metrics()

    print_model_benchmark(
        metrics_data
    )

    print(
        "\n[INFO] Generating diagnostic artifacts..."
    )

    generate_confusion_matrix(
        metrics_data
    )

    generate_model_comparison(
        metrics_data
    )

    generate_churn_distribution()

    generate_tenure_distribution()

    print(
        "\n[SUCCESS] Evaluation artifacts generated."
    )

    print(
        f"[INFO] Output directory: {FIG_DIR}"
    )


# -------------------------------------------------------------------
# Run directly
# -------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_models()
