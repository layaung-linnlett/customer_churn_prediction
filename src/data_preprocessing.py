"""Reusable data-preprocessing functions for the Customer Churn project.

These functions are the single source of truth for cleaning and feature
engineering. Notebooks 02 and 04 import and call them, and the test suite in
``tests/`` verifies their behaviour — so the exact same, validated
transformations are used everywhere instead of being copy-pasted between
notebooks.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Columns removed before modelling: a unique identifier with no predictive
# signal, and a column that is collinear with tenure x MonthlyCharges.
DROP_COLUMNS = ["customerID", "TotalCharges"]

# Columns with exactly two categories that map cleanly to 0/1 (incl. the target).
BINARY_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]


def load_raw_data(path):
    """Load the raw Telco CSV from ``path`` into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df):
    """Return a cleaned copy of ``df``.

    Fixes the one real data-quality issue in this dataset: ``TotalCharges`` is
    stored as text and hides 11 blank values. Those blanks all belong to
    ``tenure = 0`` (brand-new) customers who have not been billed yet, so ``0``
    is the correct fill value before casting the column to ``float``.

    The input DataFrame is not modified (a copy is returned).
    """
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0").astype(float)
    return df


def engineer_features(df):
    """Return a fully numeric, model-ready feature matrix from cleaned ``df``.

    Steps:
    1. Drop the ID / redundant columns (:data:`DROP_COLUMNS`).
    2. Label-encode the binary columns (:data:`BINARY_COLUMNS`) to 0/1,
       including the ``Churn`` target (No -> 0, Yes -> 1).
    3. One-hot encode the remaining multi-category columns, using
       ``drop_first=True`` to avoid the dummy-variable trap.

    The input DataFrame is not modified (a copy is returned).
    """
    df = df.copy()
    df = df.drop(columns=DROP_COLUMNS)

    encoder = LabelEncoder()
    for col in BINARY_COLUMNS:
        df[col] = encoder.fit_transform(df[col])

    df = pd.get_dummies(df, drop_first=True)
    return df


def split_features_target(df, target="Churn"):
    """Split an encoded DataFrame into features ``X`` and target ``y``."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
