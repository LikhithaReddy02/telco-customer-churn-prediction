"""
Model Factory
--------------
Creates standard scikit-learn classification models used in the
Telco Customer Churn Prediction project.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_model(name: str, random_state: int = 42):
    """
    Return a scikit-learn model based on the model name.
    """

    name = name.lower()

    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )

    elif name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=15,
            class_weight="balanced",
            random_state=random_state
        )

    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )

    elif name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown model: {name}")