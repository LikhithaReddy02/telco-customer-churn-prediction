# 📊 Telco Customer Churn Prediction & Retention Intelligence

An end-to-end machine learning project that predicts customer churn probability, classifies customer risk, identifies practical risk indicators, and provides retention recommendations through an interactive Streamlit application.

The project covers the complete ML workflow:

**Data Cleaning → Feature Engineering → Model Benchmarking → Cross-Validation → Threshold Optimization → Model Selection → Evaluation → Deployment**

---

## 🎯 Project Overview

### Business Problem

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave allows companies to take proactive retention actions.

### Objective

Build a machine learning system that:

* Predicts the probability that a customer will churn.
* Classifies customers into Low, Medium, and High Risk.
* Uses an optimized decision threshold instead of relying only on `0.50`.
* Identifies practical customer risk indicators.
* Provides a retention recommendation.
* Allows users to interact with the trained model through a Streamlit web application.

---

## 🚀 Key Features

* Automated data cleaning
* Domain-specific feature engineering
* Multiple machine learning models
* Stratified train/test split
* 5-fold stratified cross-validation
* ROC-AUC, Precision, Recall and F1 evaluation
* Probability-based threshold optimization
* Production model serialization using Joblib
* Customer-level churn probability prediction
* Rule-based risk-driver identification
* Risk-tier classification
* Retention recommendations
* Interactive Streamlit dashboard
* Diagnostic evaluation artifacts
* CLI prediction support

---

## 🧠 Machine Learning Pipeline

```text
Raw Telco Customer Data
          │
          ▼
Data Cleaning
          │
          ▼
Feature Engineering
          │
          ▼
Stratified 80/20 Train-Test Split
          │
          ▼
ColumnTransformer
 ┌────────┴─────────┐
 ▼                  ▼
Numerical         Categorical
StandardScaler    OneHotEncoder
          │
          ▼
Model Benchmarking
          │
          ▼
5-Fold Stratified Cross-Validation
          │
          ▼
Out-of-Fold Probability Prediction
          │
          ▼
F1-Based Threshold Optimization
          │
          ▼
Champion Model Selection
          │
          ▼
Final Model Training
          │
          ▼
Joblib Model Serialization
          │
          ▼
Streamlit Churn Risk Application
```

---

## 📊 Models Evaluated

The project benchmarks four classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The current training run selected **Logistic Regression** as the champion model based on the highest 5-fold cross-validation ROC-AUC.

### Current Evaluation Results

| Model               | CV ROC-AUC | Test ROC-AUC | Test F1 | Test Recall | Threshold |
| ------------------- | ---------: | -----------: | ------: | ----------: | --------: |
| Logistic Regression |     0.8620 |       0.8685 |  0.6880 |      0.7690 |      0.56 |
| Decision Tree       |     0.8356 |       0.8528 |  0.6775 |      0.7976 |      0.54 |
| Random Forest       |     0.8573 |       0.8570 |  0.6702 |      0.7548 |      0.52 |
| Gradient Boosting   |     0.8599 |       0.8623 |  0.6761 |      0.7357 |      0.37 |

### 🏆 Champion Model

**Logistic Regression**

Selection criterion:

> Highest 5-fold cross-validation ROC-AUC.

Current production configuration:

* **CV ROC-AUC:** 0.8620
* **Test ROC-AUC:** 0.8685
* **Test F1:** 0.6880
* **Test Recall:** 0.7690
* **Production threshold:** 0.56

---

## 🎚️ Decision Threshold Optimization

The model produces a churn probability rather than directly producing a business decision.

Instead of automatically using:

```text
Probability >= 0.50 → Churn
```

the project searches thresholds from `0.20` to `0.80` and selects the threshold that produces the highest F1-score using out-of-fold training probabilities.

The current champion model selected:

```text
Production Threshold = 0.56
```

Therefore:

```text
Probability >= 0.56
        ↓
Likely to Churn
```

while:

```text
Probability < 0.56
        ↓
Likely to Stay
```

This threshold is stored in:

```text
models/model_metrics.json
```

and is automatically used by the Streamlit application.

---

## 🔧 Feature Engineering

The project creates additional domain-specific features before model training.

### 1. Tenure Cohort

Customers are grouped into:

```text
0-12 months
13-24 months
25-48 months
49-72 months
```

### 2. Number of Security Services

Counts active services among:

* Online Security
* Online Backup
* Device Protection
* Tech Support

### 3. Number of Streaming Services

Counts:

* Streaming TV
* Streaming Movies

### 4. Total Active Services

Combines security and streaming services.

### 5. Automatic Payment Indicator

Identifies whether the customer uses an automatic payment method.

### 6. Charges per Active Service

Calculates:

```text
Monthly Charges / Total Active Services
```

with protection against division by zero.

### 7. Monthly-to-Total Charge Ratio

Calculates:

```text
Monthly Charges / (Total Charges + 1)
```

### 8. Senior Customer Living Alone

Identifies senior customers without a partner or dependents.

---

## 🔍 Customer Risk Drivers

The application provides practical rule-based indicators alongside the ML prediction.

Examples include:

* Month-to-month contract
* Short customer tenure
* Fiber optic internet service
* Electronic check payment
* No online security service
* No technical support service

These indicators are intended as **business-oriented risk signals**, not as direct explanations of the model's learned coefficients.

---

## 📈 Risk Classification

The application categorizes customers based on predicted churn probability:

| Probability     | Risk Tier   |
| --------------- | ----------- |
| `< 0.35`        | Low Risk    |
| `0.35 – < 0.65` | Medium Risk |
| `>= 0.65`       | High Risk   |

The churn decision itself uses the optimized production threshold of `0.56`.

Therefore, **risk tier and churn decision are intentionally separate concepts**.

---

## 💡 Retention Recommendations

The application generates a business-oriented recommendation based on risk tier.

### High Risk

Prioritize immediate retention outreach with a contract or pricing incentive.

### Medium Risk

Offer a service bundle or loyalty incentive and monitor the customer.

### Low Risk

Continue standard loyalty and customer engagement activities.

---

## 🗂️ Dataset

The project uses the **Telco Customer Churn** dataset originally published through IBM's Telco Customer Churn project.

The dataset contains approximately:

* **7,043 customer records**
* Customer demographic information
* Service information
* Contract information
* Payment information
* Monthly and total charges
* Churn target

The target variable is:

```text
Churn
```

with values:

```text
Yes
No
```

The raw dataset is intentionally excluded from Git tracking because the project contains a download script.

Dataset source:

https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

---

## 🧹 Data Cleaning

The preprocessing pipeline:

1. Loads the raw CSV.
2. Converts `TotalCharges` to numeric.
3. Handles whitespace values.
4. Assigns `0.0` TotalCharges to customers with zero tenure.
5. Removes the `customerID` identifier.
6. Handles any remaining missing `TotalCharges` values.
7. Saves the cleaned dataset locally.

---

## 🏗️ Project Structure

```text
telco-customer-churn-prediction/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   │
│   └── processed/
│       ├── cleaned_churn_data.csv
│       └── features_churn_data.csv
│
├── models/
│   ├── best_churn_model.pkl
│   ├── preprocessor.pkl
│   └── model_metrics.json
│
├── notebooks/
│   └── 01_exploration_and_modeling.ipynb
│
├── reports/
│   ├── figures/
│   └── model_performance_report.md
│
├── docs/
│   ├── PROJECT_DOCUMENTATION.md
│   └── INTERVIEW_PREP.md
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── download_data.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── predict.py
│   └── train.py
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 💻 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/telco-customer-churn-prediction.git

cd telco-customer-churn-prediction
```

### 2. Create a virtual environment

For Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, you can still run Python using the environment directly or activate it through the VS Code Python environment selector.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Download the dataset

```powershell
python src/download_data.py
```

### 5. Train the models

```powershell
python src/train.py
```

This performs:

* Data cleaning
* Feature engineering
* Train/test splitting
* Model benchmarking
* 5-fold cross-validation
* Threshold optimization
* Champion model selection
* Model serialization

### 6. Generate evaluation artifacts

```powershell
python src/evaluate.py
```

### 7. Run the Streamlit application

```powershell
streamlit run app.py
```

The application will open in your browser.

### 8. Run CLI prediction

```powershell
python src/predict.py
```

---

## 📱 Streamlit Application

The Streamlit interface allows users to enter customer information such as:

* Contract type
* Tenure
* Internet service
* Payment method
* Online security
* Technical support
* Monthly charges
* Total charges
* Customer demographic information

The application then displays:

```text
Churn Probability
        ↓
Churn Decision
        ↓
Risk Tier
        ↓
Risk Drivers
        ↓
Retention Recommendation
```

---

## 📁 Generated Model Files

### `models/best_churn_model.pkl`

Contains the complete trained scikit-learn pipeline:

```text
ColumnTransformer
       +
Champion Model
```

This allows preprocessing and prediction to remain consistent between training and inference.

### `models/preprocessor.pkl`

Contains the fitted preprocessing pipeline for inspection and reuse.

### `models/model_metrics.json`

Stores:

* Model comparison results
* Cross-validation metrics
* Test metrics
* Confusion matrix values
* Selected champion model
* Production decision threshold
* Feature information

---

## 📊 Evaluation Artifacts

The project generates diagnostic files under:

```text
reports/figures/
```

including:

* Churn distribution
* Tenure distribution
* Confusion matrix
* Model comparison
* ROC comparison
* Contract churn analysis
* Payment method analysis
* Monthly charges analysis
* Feature correlation analysis
* Feature importance visualization

---

## 🔮 Future Improvements

Potential improvements include:

1. SHAP-based model explanations for customer-level interpretability.
2. Cost-sensitive threshold optimization based on actual retention costs.
3. Automated unit testing.
4. Docker containerization.
5. GitHub Actions CI/CD.
6. Improved model monitoring.
7. Model drift detection.
8. Automated deployment.
9. More advanced hyperparameter tuning.
10. A/B testing of retention recommendations.

---

## 🎓 What I Learned

This project helped me practice an end-to-end machine learning workflow rather than only training a model.

Key areas include:

* Data preprocessing
* Feature engineering
* Classification
* Imbalanced-data handling
* Cross-validation
* Model comparison
* Probability prediction
* Decision threshold optimization
* Model serialization
* Streamlit deployment
* Git/GitHub project organization

---

## 👤 Author

**Usha Sai Likhitha Reddy**

B.Tech — Artificial Intelligence & Machine Learning

Interested in:

* Data Science
* Machine Learning
* Generative AI
* Prompt Engineering
* Data Analytics

### Connect

* GitHub: https://github.com/LikhithaReddy02
* LinkedIn: https://www.linkedin.com/in/ushasailikhitha

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
