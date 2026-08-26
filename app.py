"""Customer Churn Prediction — interactive Streamlit demo.

Enter a customer's details and get a live churn-risk prediction from the
trained XGBoost model. The same tested preprocessing functions from
``src/data_preprocessing.py`` are reused, so the demo encodes inputs exactly
the way the model was trained.

Run from the project root with:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_preprocessing import prepare_customer_for_prediction

# The tuned decision threshold chosen in notebook 07 (recall-focused).
CHURN_THRESHOLD = 0.30

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "final_model.pkl"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"


@st.cache_resource
def load_model():
    """Load the trained model once and cache it across reruns."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    """Load the cleaned dataset (used so encoding sees all categories)."""
    return pd.read_csv(CLEAN_DATA_PATH)


def main():
    st.set_page_config(
        page_title="Customer Churn Predictor",
        page_icon="📉",
        layout="wide",
    )

    st.title("📉 Customer Churn Predictor")
    st.markdown(
        "Enter a customer's details below to estimate their **risk of churning** "
        "(cancelling their service). The model is an XGBoost classifier trained on "
        "7,043 telecom customers and tuned to **catch churners early** "
        f"(it flags a customer as high-risk when churn probability ≥ "
        f"{CHURN_THRESHOLD:.0%})."
    )

    model = load_model()
    reference_df = load_reference_data()
    feature_columns = list(model.feature_names_in_)

    # ---- Input form -------------------------------------------------------
    with st.form("customer_form"):
        st.subheader("Customer details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Account**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])

        with col2:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])

        with col3:
            st.markdown("**Services**")
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox("Online security", ["No", "Yes"])
            tech_support = st.selectbox("Tech support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if not submitted:
        st.info("Fill in the form and click **Predict churn risk**.")
        return

    # ---- Build the raw customer record ------------------------------------
    # Keep service fields consistent with the dataset's conventions.
    no_internet = internet == "No"
    no_phone = phone_service == "No"

    def internet_field(value):
        """Return ``value``, or the dataset's placeholder if there is no internet."""
        return "No internet service" if no_internet else value

    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": "No phone service" if no_phone else "No",
        "InternetService": internet,
        "OnlineSecurity": internet_field(online_security),
        "OnlineBackup": internet_field("No"),
        "DeviceProtection": internet_field("No"),
        "TechSupport": internet_field(tech_support),
        "StreamingTV": internet_field(streaming_tv),
        "StreamingMovies": internet_field("No"),
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": tenure * monthly_charges,  # dropped by the pipeline
    }

    # ---- Predict ----------------------------------------------------------
    features = prepare_customer_for_prediction(customer, reference_df, feature_columns)
    churn_probability = float(model.predict_proba(features)[:, 1][0])
    is_high_risk = churn_probability >= CHURN_THRESHOLD

    # ---- Display result ---------------------------------------------------
    st.subheader("Prediction")
    result_col, gauge_col = st.columns([1, 2])

    with result_col:
        st.metric("Churn probability", f"{churn_probability:.0%}")
        if is_high_risk:
            st.error("⚠️ HIGH churn risk")
        else:
            st.success("✅ LOW churn risk")

    with gauge_col:
        st.progress(min(churn_probability, 1.0))
        st.caption(
            f"Decision threshold: {CHURN_THRESHOLD:.0%} — the model is tuned to "
            "favour catching churners over avoiding false alarms."
        )

    if is_high_risk:
        st.markdown(
            "**Recommended action:** prioritise this customer for retention — "
            "e.g. offer an incentive to move to a longer contract, or proactive "
            "outreach. The strongest churn signals in this dataset are "
            "month-to-month contracts, short tenure, high monthly charges and "
            "electronic-check payments."
        )
    else:
        st.markdown(
            "**Recommended action:** no immediate retention spend needed — this "
            "customer looks stable. Continue standard engagement."
        )

    with st.expander("How does this work?"):
        st.markdown(
            "Your inputs are encoded with the **same tested preprocessing "
            "pipeline** (`src/data_preprocessing.py`) used to train the model, "
            "then scored by an **XGBoost + SMOTE** classifier. On the held-out "
            "test set this model catches **77% of real churners** (recall 0.77). "
            "See the notebooks for the full methodology."
        )


if __name__ == "__main__":
    main()
