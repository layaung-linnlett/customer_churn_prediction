"""
setup.py — Regenerate all model artefacts from scratch.

Run this script if any .pkl file is missing:
    python setup.py

It will:
1. Load and clean the raw dataset
2. Engineer all features
3. Train the final XGBoost model
4. Save all artefacts to models/
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("Customer Churn Prediction — Setup")
print("=" * 55)

# ── Check dataset exists ──────────────────────────────────────────────────────
csv_candidates = [
    'data/telco_churn.csv',
    'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
]

df = None
for path in csv_candidates:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\nDataset loaded: {path}")
        print(f"Shape: {df.shape}")
        break

if df is None:
    print("\nERROR: Dataset not found.")
    print("Please download telco_churn.csv from Kaggle and place it in data/")
    print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    sys.exit(1)

# ── Create output folders ─────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
print("\nOutput folders ready.")

# ── Phase 1: Clean ────────────────────────────────────────────────────────────
print("\n[1/5] Cleaning data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop(columns=['customerID', 'TotalCharges'], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"      {df.shape[0]} rows, {df.shape[1]} columns")

# ── Phase 3: Feature engineering ─────────────────────────────────────────────
print("\n[2/5] Engineering features...")

def tenure_group(t):
    if t <= 12:   return 'new'
    elif t <= 24: return 'developing'
    elif t <= 48: return 'established'
    else:         return 'loyal'

df['tenure_group']      = df['tenure'].apply(tenure_group)
service_cols            = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies']
df['num_services']      = df[service_cols].apply(
                              lambda r: (r == 'Yes').sum(), axis=1)
df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract',
                  'PaymentMethod', 'tenure_group']
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
print(f"      {df.shape[1]} features after encoding")

# ── Phase 3: Split and scale ──────────────────────────────────────────────────
print("\n[3/5] Splitting and scaling...")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler   = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'charge_per_tenure', 'num_services']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv',   index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv',   index=False)
print(f"      Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

# ── Phase 4: SMOTE + Train ────────────────────────────────────────────────────
print("\n[4/5] Training model...")
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

spw   = round((y_train == 0).sum() / (y_train == 1).sum(), 2)
model = XGBClassifier(
    scale_pos_weight=spw,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_sm, y_train_sm)

from sklearn.metrics import f1_score, recall_score, roc_auc_score
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.45).astype(int)
print(f"      F1={f1_score(y_test,y_pred)*100:.1f}%  "
      f"Recall={recall_score(y_test,y_pred)*100:.1f}%  "
      f"AUC={roc_auc_score(y_test,y_prob):.3f}")

# ── Phase 7: SHAP ─────────────────────────────────────────────────────────────
print("\n[5/5] Creating SHAP explainer...")
import shap
explainer = shap.TreeExplainer(model)

# ── Save all artefacts ────────────────────────────────────────────────────────
feature_cols = list(X_train.columns)

artefacts = {
    'models/best_model.pkl':     model,
    'models/scaler.pkl':         scaler,
    'models/shap_explainer.pkl': explainer,
    'models/feature_cols.pkl':   feature_cols,
    'models/evaluation_summary.pkl': {
        'model_name':     'XGBoost (default)',
        'best_threshold':  0.45,
        'recall':          recall_score(y_test, y_pred),
        'precision':       __import__('sklearn.metrics',
                               fromlist=['precision_score']
                               ).precision_score(y_test, y_pred),
        'f1':              f1_score(y_test, y_pred),
        'roc_auc':         roc_auc_score(y_test, y_prob)
    }
}

print("\nSaving artefacts:")
for path, obj in artefacts.items():
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    size = os.path.getsize(path)
    print(f"  {path:<40} {size/1024:.1f} KB")

print("\n" + "=" * 55)
print("Setup complete. All artefacts saved to models/")
print("Run the app with: streamlit run app/streamlit_app.py")
print("=" * 55)
