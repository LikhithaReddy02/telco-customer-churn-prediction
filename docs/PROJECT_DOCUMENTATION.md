# Deep-Dive Project Documentation: Telco Customer Churn System

## 1. Problem Statement & Motivation
Subscription businesses face significant customer attrition. In telecom, contracts often transition to month-to-month terms, creating friction points where dissatisfied customers switch providers. This project operationalizes machine learning to forecast churn risk before contracts lapse.

## 2. Preprocessing & Data Cleaning Decisions
1. **TotalCharges Whitespaces:**
   - *Problem:* 11 records have blank spaces for `TotalCharges`.
   - *Decision:* Imputed with `0.0`.
   - *Reason:* All 11 records have `tenure == 0` (new enrollments who have not completed their first billing cycle). Dropping them causes data loss; mean imputation injects synthetic bias.
2. **High-Cardinality Identifiers:**
   - *Problem:* `customerID` has 7,043 unique alphanumeric strings.
   - *Decision:* Dropped prior to modeling.
   - *Reason:* Prevents memory leakage and spurious memorization.
3. **Categorical Encodings:**
   - *Binary features* mapped to 0/1.
   - *Multi-category features* converted via one-hot encoding with `drop_first=True` to avoid multicollinearity (dummy variable trap).
4. **Standardization:**
   - Standardized `tenure`, `MonthlyCharges`, and `TotalCharges` using train-set means and standard deviations, preventing test-set data leakage.

## 3. Exploratory Data Analysis Takeaways
- **Contract Impact:** Month-to-month customers exhibit >40% churn, while 1-year and 2-year contracts show <12% churn.
- **Tenure Curve:** Churn is heavily front-loaded in the first 12 months.
- **Internet Infrastructure:** Fiber optic users churn at higher rates, correlating with higher monthly bills and potential service disruptions.
- **Payment Method Friction:** Electronic check users have the highest attrition, whereas automated bank/card payments promote longevity.
