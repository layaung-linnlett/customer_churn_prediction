# customer_churn_prediction

A machine learning project to predict which telecom customers are likely to churn,
enabling the business to take proactive retention action.

**Final Model:** XGBoost + SMOTE + Threshold Tuning  
**Final Recall:** 75% (baseline: 58%)  
**Business Goal:** Catch at least 70% of churners before they leave

## Project Structure

customer-churn/
├── data/
│   ├── Telco-Customer-Churn.csv       ← raw dataset
│   └── cleaned_data.csv               ← processed dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_building.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_model_improvement.ipynb
│   └── 08_final_model.ipynb
├── models/
│   └── final_model.pkl                ← saved XGBoost model
├── outputs/                           ← plots and visualizations
└── README.md

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** `Churn` (Yes/No) — binary classification
- **Key features:** tenure, MonthlyCharges, Contract type, PaymentMethod, InternetService

- ## Methodology

1. **Data Exploration** — checked shape, data types, missing values, class imbalance (73% No / 27% Yes)
2. **Data Cleaning** — replaced 11 hidden blank values in `TotalCharges`, fixed data type from str to float
3. **EDA** — visualized churn patterns across Contract, MonthlyCharges, PaymentMethod, and tenure
4. **Feature Engineering** — dropped redundant features, Label Encoded binary columns, One-Hot Encoded categorical columns
5. **Baseline Model** — Logistic Regression (Recall: 0.58)
6. **Model Improvement** — tested Random Forest, XGBoost, applied SMOTE for class imbalance, tuned prediction threshold to 0.3
7. **Final Model** — XGBoost + SMOTE + threshold 0.3 (Recall: 0.75)

## Results

| Model | Recall | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression (baseline) | 0.58 | 0.63 | 82% |
| Random Forest | 0.47 | 0.54 | 79% |
| Random Forest + SMOTE | 0.60 | 0.58 | 77% |
| XGBoost | 0.52 | 0.57 | 79% |
| XGBoost + SMOTE | 0.62 | 0.60 | 78% |
| **XGBoost + SMOTE + Threshold 0.3** | **0.75** | **0.59** | **72%** |

**Key insight:** Accuracy alone is misleading for imbalanced datasets.
The final model sacrifices some accuracy to catch 75% of real churners —
a deliberate business decision to minimize missed churners over false alarms.

## How to Run

1. Clone this repository
2. Install dependencies:
pip install -r requirements.txt


3. Open notebooks in order starting from `01_data_exploration.ipynb`
4. Final model is saved in `models/final_model.pkl`

## Technologies Used

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn, XGBoost
- **Imbalanced data:** imbalanced-learn (SMOTE)
- **Model persistence:** Joblib
