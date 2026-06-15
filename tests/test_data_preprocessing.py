"""Unit tests for src/data_preprocessing.py.

Run from the project root with:  python -m pytest
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    clean_data,
    engineer_features,
    load_raw_data,
    prepare_customer_for_prediction,
    split_features_target,
)

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "telco-customer-churn.csv"
)


def make_sample_df():
    """Build a tiny DataFrame that mirrors the real dataset's structure.

    Row 3 has a blank TotalCharges (tenure 0) to mimic the real data trap.
    """
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3"],
            "gender": ["Female", "Male", "Female"],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "PhoneService": ["No", "Yes", "Yes"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "tenure": [1, 34, 0],
            "MonthlyCharges": [29.85, 56.95, 20.0],
            "TotalCharges": ["29.85", "1889.5", " "],  # blank for the tenure-0 row
            "Churn": ["No", "Yes", "No"],
        }
    )


# ---- clean_data -----------------------------------------------------------

def test_clean_data_converts_totalcharges_to_float():
    cleaned = clean_data(make_sample_df())
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


def test_clean_data_fills_blank_with_zero():
    cleaned = clean_data(make_sample_df())
    # The single blank row should now be 0.0
    assert (cleaned["TotalCharges"] == 0.0).sum() == 1
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_data_does_not_mutate_input():
    original = make_sample_df()
    _ = clean_data(original)
    # The original is untouched: still non-numeric and still holding the blank
    assert not pd.api.types.is_numeric_dtype(original["TotalCharges"])
    assert (original["TotalCharges"] == " ").any()


# ---- engineer_features ----------------------------------------------------

def test_engineer_features_drops_id_and_totalcharges():
    encoded = engineer_features(clean_data(make_sample_df()))
    assert "customerID" not in encoded.columns
    assert "TotalCharges" not in encoded.columns


def test_engineer_features_is_all_numeric():
    encoded = engineer_features(clean_data(make_sample_df()))
    # bool dummy columns count as numeric for pandas / sklearn
    assert all(pd.api.types.is_numeric_dtype(encoded[c]) for c in encoded.columns)


def test_engineer_features_encodes_churn_to_binary():
    encoded = engineer_features(clean_data(make_sample_df()))
    assert set(encoded["Churn"].unique()) <= {0, 1}


def test_engineer_features_does_not_mutate_input():
    cleaned = clean_data(make_sample_df())
    before_cols = list(cleaned.columns)
    _ = engineer_features(cleaned)
    assert list(cleaned.columns) == before_cols


# ---- split_features_target ------------------------------------------------

def test_split_features_target():
    encoded = engineer_features(clean_data(make_sample_df()))
    X, y = split_features_target(encoded)
    assert "Churn" not in X.columns
    assert y.name == "Churn"
    assert len(X) == len(y) == 3


# ---- prepare_customer_for_prediction --------------------------------------

def test_prepare_customer_matches_feature_columns():
    reference = clean_data(make_sample_df())
    feature_columns = list(engineer_features(reference).drop(columns=["Churn"]).columns)

    customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "PhoneService": "Yes", "PaperlessBilling": "Yes",
        "Contract": "Two year", "InternetService": "Fiber optic",
        "tenure": 12, "MonthlyCharges": 80.0,
    }
    encoded = prepare_customer_for_prediction(customer, reference, feature_columns)

    assert list(encoded.columns) == feature_columns
    assert encoded.shape[0] == 1
    assert all(pd.api.types.is_numeric_dtype(encoded[c]) for c in encoded.columns)


def test_prepare_customer_sets_correct_dummy():
    reference = clean_data(make_sample_df())
    feature_columns = list(engineer_features(reference).drop(columns=["Churn"]).columns)

    customer = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "PhoneService": "Yes", "PaperlessBilling": "No",
        "Contract": "Two year", "InternetService": "DSL",
        "tenure": 5, "MonthlyCharges": 50.0,
    }
    encoded = prepare_customer_for_prediction(customer, reference, feature_columns)
    # The 'Two year' contract dummy must be switched on
    assert encoded["Contract_Two year"].iloc[0] == 1.0


# ---- integration on the real dataset (skipped if the file is absent) ------

@pytest.mark.skipif(
    not os.path.exists(RAW_DATA_PATH), reason="raw dataset not available"
)
def test_full_pipeline_on_real_data():
    df = engineer_features(clean_data(load_raw_data(RAW_DATA_PATH)))
    # 7,043 customers, 29 features + the Churn target = 30 columns
    assert df.shape == (7043, 30)
    assert df["Churn"].isin([0, 1]).all()
    # No missing values survive the pipeline
    assert df.isna().sum().sum() == 0
