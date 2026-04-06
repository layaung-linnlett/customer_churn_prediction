# notebooks/

Run these notebooks in order. Each builds on the outputs of the previous one.

## Notebook guide

| # | Notebook | Key outputs | Est. runtime |
|---|---|---|---|
| 01 | `01_eda.ipynb` | EDA plots, insight summary | 1–2 min |
| 02 | `02_modelling.ipynb` | X_train/test splits, scaler.pkl, best_model.pkl | 2–4 min |
| 03 | `03_model_evaluation.ipynb` | ROC curve, PR curve, threshold analysis | 1–2 min |
| 04 | `04_hyperparameter_tuning.ipynb` | Final best_model.pkl, evaluation_summary.pkl | 5–8 min |
| 05 | `05_shap_explainability.ipynb` | shap_explainer.pkl, shap_importance.csv | 1–2 min |

## Key decisions documented in notebooks

- **`01_eda.ipynb`**: Class imbalance discovered (73/27 split) —
  why accuracy is a misleading metric for churn prediction and why
  recall and F1 are used instead

- **`02_modelling.ipynb`**: Why `drop_first=True` prevents the dummy
  variable trap; why the scaler is fitted on training data only to
  prevent data leakage; why SMOTE is applied after splitting not before;
  why threshold 0.45 was chosen over default 0.50 — the business cost
  asymmetry between false negatives and false positives

- **`03_model_evaluation.ipynb`**: ROC curve vs Precision-Recall curve —
  why PR curve is more honest for imbalanced data; full confusion matrix
  breakdown translated into business language

- **`04_hyperparameter_tuning.ipynb`**: Why the tuned model was rejected
  despite 85.32% CV F1 — overfitting to SMOTE synthetic patterns caused
  worse real-world performance; why test set validation always overrules
  cross-validation scores

- **`05_shap_explainability.ipynb`**: Why two engineered features
  (`charge_per_tenure` and `num_services`) ranked above all original
  dataset columns in SHAP importance, validating the feature engineering
  decisions made in modelling; correlation check confirming tenure and
  tenure_group_established capture genuinely different signals (r=0.12)
## Prerequisites

```bash
pip install -r requirements.txt
```

Download `telco_churn.csv` from Kaggle and place in `data/` before running.
