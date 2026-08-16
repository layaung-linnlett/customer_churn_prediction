# Customer Churn Prediction

A model that flags the telecom customers most likely to cancel, catching 77% of real churners so the retention team has a shortlist to work instead of a whole customer base.

**Live demo:** https://customer-churn-predictor-telecom.streamlit.app

## Key Findings

- **27% of the 7,043 customers churned.** That's the base rate everything else is measured against.
- **A standard model looked fine and wasn't.** Logistic regression scored 82% accuracy but caught only 58% of churners. Accuracy hid the weakness, because predicting "nobody churns" alone scores 73% on this data.
- **The final model catches 77% of churners, up from 58%**, using SMOTE plus a decision threshold lowered to 0.3. It clears the 70% target I set at the start.
- **That gain cost accuracy**, which fell from 82% to 73%. I took the trade deliberately: a missed churner loses a whole contract, a false alarm costs one retention offer.
- **Electronic-check payment is the strongest single driver in the model** (feature importance 0.22), ahead of having no internet service (0.16) and being on a two-year contract (0.09).
- **In the EDA, contract length separates churners most cleanly** — month-to-month customers churn far more than one- or two-year customers. It ranks below payment method in the model because the one-hot contract columns split that signal across three features.
- **Churn concentrates in the first few months.** Low-tenure customers dominate, and monthly charges around $70–$110 are the highest-risk band.

## Screenshots

**Top drivers of churn in the final XGBoost model**
![Feature importance](outputs/figures/feature_importance.png)

**Churn by contract type — the clearest split in the EDA**
![Churn by contract type](outputs/figures/churn_by_contract.png)

**Churn by tenure — risk sits in the first months**
![Churn by tenure](outputs/figures/churn_by_tenure.png)

**Final confusion matrix — far fewer missed churners**
![Final confusion matrix](outputs/figures/confusion_matrix_final.png)

**The Streamlit app** ([`app.py`](app.py)) takes a customer's details and returns a live risk score from the trained model, reusing the same tested preprocessing functions as the notebooks.

![Customer Churn Predictor demo](outputs/figures/app_demo.png)

<details>
<summary>More charts</summary>

- `outputs/figures/churn_by_monthly_charges.png`
- `outputs/figures/churn_by_payment_method.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/confusion_matrix_baseline.png`

</details>

## Tech Stack

| Tool | Used for |
|---|---|
| Python 3.11+ | Core language |
| pandas / NumPy | Loading, cleaning and reshaping the data |
| Matplotlib / Seaborn | EDA and result charts |
| scikit-learn | Train/test split, logistic regression, random forest, metrics |
| XGBoost | The final classifier |
| imbalanced-learn (SMOTE) | Balancing the training set |
| joblib | Saving and reloading the trained model |
| Jupyter Notebook | The eight analysis notebooks |
| pytest | Unit tests for the preprocessing module |
| Streamlit | The interactive demo app |

## Methodology

1. **Data exploration** — 7,043 rows × 21 columns. `TotalCharges` was stored as text and hid 11 blank values.
2. **Cleaning** — converted `TotalCharges` to numeric and set those 11 blanks to `0`; they all belong to brand-new customers with `tenure = 0`. Saved one clean dataset that everything downstream reuses.
3. **EDA** — plotted churn against contract type, monthly charges, payment method and tenure before touching a model, to know what a sensible result should look like.
4. **Feature engineering** — dropped `customerID` (no signal) and `TotalCharges` (collinear with tenure × charges), label-encoded the binary columns and one-hot-encoded the multi-category ones.
5. **Baseline** — logistic regression, for an honest and interpretable starting point. Recall 0.58.
6. **Improvement** — compared random forest and XGBoost, applied SMOTE to the training data only, then lowered the decision threshold to 0.3.
7. **Final model** — XGBoost + SMOTE + 0.3 threshold, saved with joblib and verified by reloading it.

Logistic regression sets the baseline. Tree models pick up interactions it can't. SMOTE targets the imbalance itself rather than patching the symptom, and the threshold is where the business cost actually enters the model — 0.3 is the point where recall crossed the 70% target.

The cleaning and encoding logic lives in [`src/data_preprocessing.py`](src/data_preprocessing.py) as tested functions rather than notebook cells. Notebooks 02 and 04 import and call them, so the same transformations run everywhere and the [`tests/`](tests/) suite covers them.

### Results

| Model | Recall (churn) | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression (baseline) | 0.58 | 0.63 | 82% |
| Random Forest | 0.47 | 0.54 | 79% |
| Random Forest + SMOTE | 0.61 | 0.59 | 77% |
| XGBoost | 0.53 | 0.57 | 79% |
| XGBoost + SMOTE | 0.61 | 0.58 | 77% |
| **XGBoost + SMOTE + Threshold 0.3** | **0.77** | **0.60** | **73%** |

## Project Structure

```
customer_churn_prediction/
├── app.py                               # Interactive Streamlit demo app
├── data/
│   ├── raw/
│   │   └── telco-customer-churn.csv     # Original dataset — never modified
│   └── processed/
│       ├── telco_churn_clean.csv        # Cleaned data (output of notebook 02)
│       └── model_ready.csv              # Encoded model matrix (output of notebook 04)
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Shape, data types, hidden missing values, class imbalance
│   ├── 02_data_cleaning.ipynb           # Fix TotalCharges, handle blanks, save clean data
│   ├── 03_eda.ipynb                     # Churn patterns by contract, charges, tenure, payment
│   ├── 04_feature_engineering.ipynb     # Drop redundant columns, encode categoricals
│   ├── 05_model_building.ipynb          # Logistic Regression baseline
│   ├── 06_model_evaluation.ipynb        # Confusion matrix, precision/recall/F1, why recall matters
│   ├── 07_model_improvement.ipynb       # Random Forest, XGBoost, SMOTE, threshold tuning
│   └── 08_final_model.ipynb             # Final model, feature importance, save & verify
├── outputs/
│   ├── figures/                         # All charts saved by the notebooks
│   └── models/
│       └── final_model.pkl              # Trained, ready-to-reuse final model
├── src/
│   ├── __init__.py
│   └── data_preprocessing.py            # Reusable, tested cleaning & encoding functions
├── tests/
│   └── test_data_preprocessing.py       # Unit tests for the preprocessing module
├── pytest.ini
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## How To Run

```bash
git clone https://github.com/layaung-linnlett/customer_churn_prediction.git
cd customer_churn_prediction

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Run the notebooks in numerical order — each one depends on the file the previous one wrote. Notebook 02 produces the cleaned data, 04 produces the model-ready matrix, 08 saves the final model.

```bash
jupyter notebook          # then run 01 → 08

python -m pytest          # unit tests for the preprocessing module
streamlit run app.py      # the demo app
```

The trained model loads on its own if you'd rather not re-run everything:

```python
import joblib
model = joblib.load("outputs/models/final_model.pkl")
# Predict a churner when probability >= 0.3 (the tuned threshold)
churn_flags = (model.predict_proba(X)[:, 1] >= 0.3).astype(int)
```

## Limitations & Future Work

**Limitations**

- The dataset is one static snapshot, so there's no seasonality and no way to see behaviour change over time.
- SMOTE invents synthetic minority rows. They're plausible, not real, and the model is partly fitted to them.
- Hyperparameters were only lightly tuned — no grid search or Bayesian search.
- Higher recall means lower precision. Some of the retention budget will go to customers who were never going to leave.
- The 0.3 threshold was chosen to hit a 70% recall target, not from real campaign economics. With actual costs for a retention offer and a lost contract, the optimal threshold would probably land somewhere else.

**Future work**

- Systematic hyperparameter tuning with cross-validation for more stable estimates.
- More engineered features: tenure buckets, services per customer, charge ratios.
- Calibrate the predicted probabilities and set the threshold from real campaign costs.
- Deploy as an API and monitor for drift.

## Contact

**La Yaung Linn Lett**
- GitHub: [github.com/layaung-linnlett](https://github.com/layaung-linnlett)
- LinkedIn: [linkedin.com/in/layaung-linnlett](https://www.linkedin.com/in/layaung-linnlett/)
- Email: layaunglinnlett1@gmail.com

*Dataset: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features).*
