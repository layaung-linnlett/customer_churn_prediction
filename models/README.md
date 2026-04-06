# models/

This folder contains all trained model artefacts saved as `.pkl` files.

## Files

| File | Size (approx) | Description |
|---|---|---|
| `best_model.pkl` | ~315 KB | Trained XGBoost classifier (default settings) |
| `scaler.pkl` | ~1 KB | Fitted StandardScaler (mean/std from training data only) |
| `shap_explainer.pkl` | ~1 MB | SHAP TreeExplainer for local explanations |
| `feature_cols.pkl` | ~1 KB | Ordered list of feature column names |
| `evaluation_summary.pkl` | ~1 KB | Final metrics dictionary |

## Important notes

### Why the scaler is saved separately
The scaler was fitted on training data only — fitting on all data before
splitting would cause data leakage. The saved scaler ensures the Streamlit
app applies the exact same mean and std transformation to new customer inputs
that was used during training. Using a different scaler at prediction time
would produce incorrect, inconsistent results.

### Why feature_cols is saved
The model expects features in a specific order and with specific column names
(after one-hot encoding). Saving the column list ensures the Streamlit app
builds its input DataFrame in the exact same structure the model was trained on.

### Final model choice
XGBoost with default settings at threshold 0.45 was selected over a tuned
model because hyperparameter tuning caused overfitting to SMOTE synthetic
patterns, resulting in worse real test set performance despite a higher
cross-validated score.

## Regenerating model files
If any file is missing, run notebooks 03 through 07 in order.
The save commands at the end of each notebook will recreate all artefacts.
