## 🚀 Comprehensive Credit Risk Assessment Framework: A Machine Learning Approach to Loan Default Prediction
### Project Overview

This project implements a robust machine learning framework for predicting loan default (credit risk) using the lending_data.csv dataset. The goal is to develop a predictive model that accurately identifies high-risk loans to mitigate financial losses for the lending institution.

### The framework consists of two primary machine learning tasks:

Regression Task (Borrower Income Prediction): Training a model to estimate borrower_income as an intermediate feature.

Classification Task (Loan Default Prediction): Training and evaluating various classifiers to predict the binary loan_status (0: Healthy Loan, 1: High-Risk Loan/Default).



## 📊 Key Findings
Class Imbalance: The dataset exhibited a severe class imbalance, with 99.64% of loans being healthy (0) and only 0.36% being high-risk (1). Addressing this using techniques like class_weight='balanced' and BalancedRandomForestClassifier was critical.

Best Classifier: The Balanced Random Forest Classifier emerged as the top-performing model, demonstrating superior ability to handle the imbalanced nature of the data.

Balanced Accuracy: 0.9996

ROC-AUC Score: 0.9999

Feature Importance: The most critical features for predicting loan default were loan_to_income_ratio, total_debt, and borrower_income.

Business Impact: Based on the cost-benefit analysis (assuming a default cost of $100,000 and a False Positive cost of $1,000), the model demonstrated a potential estimated saving of $49,825,000.00 in the test set by correctly identifying high-risk loans.


## 🛠 Prerequisites

The project requires the following libraries, all of which were imported in the initial code block:

Python (3.8+)

numpy

pandas

matplotlib

seaborn

plotly

scikit-learn (sklearn)

imblearn (for handling imbalance)

lightgbm

## ⚙️ Methodology

1. Data Loading and Exploration
Loaded the lending_data.csv file.

Confirmed dataset shape: 77,536 rows and 8 columns.

Verified no missing values were present.

Identified the severe class imbalance in the loan_status target variable.

## 2. Data Preprocessing and Feature Engineering  

Outlier Handling: Outliers in numerical features were handled using the Interquartile Range (IQR) Capping method.

Feature Engineering: Four new, highly informative financial ratio features were created:

income_to_debt_ratio

loan_to_income_ratio

interest_to_income_ratio

credit_utilization

## 3. Task 1: Borrower Income Prediction (Regression)

Goal: Predict borrower_income to potentially enhance the classification task.

Technique: Log transformation was applied to the target variable to reduce skewness.

Best Model: Random Forest Regressor achieved the highest performance.

R² Score: 0.9946

MAE: $1,675.29 (Mean Absolute Error)

## 4. Task 2: Loan Default Prediction (Classification)

Model Selection: Multiple classifiers were trained, including Logistic Regression, Random Forest, XGBoost, and Balanced Random Forest.

Imbalance Handling: Models were configured to handle imbalance through methods like class_weight='balanced' (for LogReg, RF) or scale_pos_weight (for XGBoost), and Balanced Random Forest (which subsamples the majority class).

Evaluation Metric: Balanced Accuracy and ROC-AUC were prioritized over standard accuracy due to the high class imbalance.
