# Model Performance Benchmark Report

## 1. Champion Model Selection
The winning architecture is **Random Forest Ensemble**, delivering balanced discrimination on held-out customer accounts:
- **Test ROC-AUC**: 0.8712
- **Test F1-Score**: 0.6928
- **Test Recall**: 0.6952
- **Test Precision**: 0.6903
- **Test Accuracy**: 0.8162

## 2. Multi-Model Benchmark Comparison Table
| Model Architecture | 5-Fold CV ROC-AUC | Test Accuracy | Test Recall | Test F1-Score | Test ROC-AUC | Training Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression (L2) | 0.8507 | 0.8055 | 0.6048 | 0.6496 | 0.8657 | 0.266 |
| Decision Tree (Gini) | 0.8537 | 0.8190 | 0.6333 | 0.6760 | 0.8687 | 0.263 |
| Random Forest Ensemble | 0.8562 | 0.8162 | 0.6952 | 0.6928 | 0.8712 | 5.080 |
| Gradient Boosting Classifier | 0.8262 | 0.7019 | 0.0000 | 0.0000 | 0.8412 | 2.164 |

## 3. Decision Threshold Optimization
| Decision Threshold | Precision | Recall | F1-Score | Strategic Business Application |
|:---:|:---:|:---:|:---:|:---|
| 0.25 | 0.5399 | 0.9024 | 0.6756 | Aggressive Retention (Maximize Customer Capture) |
| 0.30 | 0.5723 | 0.8762 | 0.6924 | Aggressive Retention (Maximize Customer Capture) |
| 0.35 | 0.6131 | 0.8262 | 0.7039 | Aggressive Retention (Maximize Customer Capture) |
| 0.40 | 0.6338 | 0.7952 | 0.7054 | Balanced Default |
| 0.45 | 0.6517 | 0.7619 | 0.7025 | Balanced Default |
| 0.50 | 0.6903 | 0.6952 | 0.6928 | Balanced Default |
| 0.55 | 0.7297 | 0.5786 | 0.6454 | Budget Constrained (High Probability Targets) |
| 0.60 | 0.7706 | 0.4238 | 0.5469 | Budget Constrained (High Probability Targets) |
