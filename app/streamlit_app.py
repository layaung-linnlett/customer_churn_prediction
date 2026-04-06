# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Absolute path to models folder ───────────────────────────────────────────
MODELS_DIR = r"C:\Users\l2-lett\OneDrive - UWE Bristol\Customer_Churn_Prediction\models"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='📊',
    layout='wide'
)

# ── Load model, scaler, explainer ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'shap_explainer.pkl'), 'rb') as f:
        explainer = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, explainer, feature_cols

model, scaler, explainer, feature_cols = load_artifacts()

# ── Helper: build input dataframe from user selections ────────────────────────
def build_input(inputs):
    """
    Takes the raw user inputs from the sidebar and returns a
    one-row dataframe matching the exact feature columns the
    model was trained on — same encoding, same scaling.
    """
    row = {col: 0 for col in feature_cols}

    # Numerical features — scale using saved scaler
    num_cols = ['tenure', 'MonthlyCharges',
                'charge_per_tenure', 'num_services']

    raw_num = pd.DataFrame([[
        inputs['tenure'],
        inputs['MonthlyCharges'],
        inputs['MonthlyCharges'] / (inputs['tenure'] + 1),  # charge_per_tenure
        inputs['num_services']
    ]], columns=num_cols)

    scaled = scaler.transform(raw_num)
    row['tenure']            = scaled[0][0]
    row['MonthlyCharges']    = scaled[0][1]
    row['charge_per_tenure'] = scaled[0][2]
    row['num_services']      = scaled[0][3]

    # Binary label-encoded features
    row['gender']           = 1 if inputs['gender'] == 'Male' else 0
    row['Partner']          = 1 if inputs['Partner'] == 'Yes' else 0
    row['Dependents']       = 1 if inputs['Dependents'] == 'Yes' else 0
    row['PhoneService']     = 1 if inputs['PhoneService'] == 'Yes' else 0
    row['PaperlessBilling'] = 1 if inputs['PaperlessBilling'] == 'Yes' else 0

    # One-hot: MultipleLines
    if inputs['MultipleLines'] == 'Yes':
        row['MultipleLines_Yes'] = 1
    elif inputs['MultipleLines'] == 'No phone service':
        row['MultipleLines_No phone service'] = 1

    # One-hot: InternetService
    if inputs['InternetService'] == 'Fiber optic':
        row['InternetService_Fiber optic'] = 1
    elif inputs['InternetService'] == 'No':
        row['InternetService_No'] = 1

    # One-hot: OnlineSecurity
    if inputs['OnlineSecurity'] == 'Yes':
        row['OnlineSecurity_Yes'] = 1
    elif inputs['OnlineSecurity'] == 'No internet service':
        row['OnlineSecurity_No internet service'] = 1

    # One-hot: OnlineBackup
    if inputs['OnlineBackup'] == 'Yes':
        row['OnlineBackup_Yes'] = 1
    elif inputs['OnlineBackup'] == 'No internet service':
        row['OnlineBackup_No internet service'] = 1

    # One-hot: DeviceProtection
    if inputs['DeviceProtection'] == 'Yes':
        row['DeviceProtection_Yes'] = 1
    elif inputs['DeviceProtection'] == 'No internet service':
        row['DeviceProtection_No internet service'] = 1

    # One-hot: TechSupport
    if inputs['TechSupport'] == 'Yes':
        row['TechSupport_Yes'] = 1
    elif inputs['TechSupport'] == 'No internet service':
        row['TechSupport_No internet service'] = 1

    # One-hot: StreamingTV
    if inputs['StreamingTV'] == 'Yes':
        row['StreamingTV_Yes'] = 1
    elif inputs['StreamingTV'] == 'No internet service':
        row['StreamingTV_No internet service'] = 1

    # One-hot: StreamingMovies
    if inputs['StreamingMovies'] == 'Yes':
        row['StreamingMovies_Yes'] = 1
    elif inputs['StreamingMovies'] == 'No internet service':
        row['StreamingMovies_No internet service'] = 1

    # One-hot: Contract
    if inputs['Contract'] == 'One year':
        row['Contract_One year'] = 1
    elif inputs['Contract'] == 'Two year':
        row['Contract_Two year'] = 1

    # One-hot: PaymentMethod
    pm = inputs['PaymentMethod']
    if pm == 'Credit card (automatic)':
        row['PaymentMethod_Credit card (automatic)'] = 1
    elif pm == 'Electronic check':
        row['PaymentMethod_Electronic check'] = 1
    elif pm == 'Mailed check':
        row['PaymentMethod_Mailed check'] = 1

    # Tenure group
    t = inputs['tenure']
    if 12 < t <= 24:
        if 'tenure_group_developing' in row:
            row['tenure_group_developing'] = 1
    elif 24 < t <= 48:
        if 'tenure_group_established' in row:
            row['tenure_group_established'] = 1
    elif t > 48:
        if 'tenure_group_loyal' in row:
            row['tenure_group_loyal'] = 1

    return pd.DataFrame([row])[feature_cols]

# ── UI: Header ────────────────────────────────────────────────────────────────
st.title('Customer Churn Predictor')
st.markdown(
    'Enter customer details in the sidebar. The model will predict '
    'their churn risk and explain the key drivers using SHAP.'
)
st.divider()

# ── UI: Sidebar inputs ────────────────────────────────────────────────────────
st.sidebar.header('Customer details')
st.sidebar.markdown('Fill in all fields then click **Predict**.')

with st.sidebar:
    st.subheader('Demographics')
    gender      = st.selectbox('Gender',         ['Male', 'Female'])
    partner     = st.selectbox('Has partner',    ['Yes', 'No'])
    dependents  = st.selectbox('Has dependents', ['Yes', 'No'])

    st.subheader('Account')
    tenure       = st.slider('Tenure (months)', 0, 72, 12)
    contract     = st.selectbox('Contract type',
                                ['Month-to-month', 'One year', 'Two year'])
    paperless    = st.selectbox('Paperless billing', ['Yes', 'No'])
    payment      = st.selectbox('Payment method', [
                                'Electronic check', 'Mailed check',
                                'Bank transfer (automatic)',
                                'Credit card (automatic)'])
    monthly      = st.slider('Monthly charges (£)', 18.0, 120.0, 65.0, 0.5)

    st.subheader('Services')
    phone        = st.selectbox('Phone service', ['Yes', 'No'])
    multi_lines  = st.selectbox('Multiple lines',
                                ['No', 'Yes', 'No phone service'])
    internet     = st.selectbox('Internet service',
                                ['DSL', 'Fiber optic', 'No'])
    online_sec   = st.selectbox('Online security',
                                ['No', 'Yes', 'No internet service'])
    online_bkp   = st.selectbox('Online backup',
                                ['No', 'Yes', 'No internet service'])
    device_prot  = st.selectbox('Device protection',
                                ['No', 'Yes', 'No internet service'])
    tech_sup     = st.selectbox('Tech support',
                                ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV',
                                ['No', 'Yes', 'No internet service'])
    streaming_mv = st.selectbox('Streaming movies',
                                ['No', 'Yes', 'No internet service'])

    predict_btn  = st.button('Predict churn risk', type='primary',
                              use_container_width=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if predict_btn:
    # Count services
    services = [online_sec, online_bkp, device_prot,
                tech_sup, streaming_tv, streaming_mv]
    num_services = sum(1 for s in services if s == 'Yes')

    inputs = {
        'gender': gender, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone,
        'MultipleLines': multi_lines, 'InternetService': internet,
        'OnlineSecurity': online_sec, 'OnlineBackup': online_bkp,
        'DeviceProtection': device_prot, 'TechSupport': tech_sup,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_mv,
        'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly,
        'num_services': num_services
    }

    # Build feature row and predict
    X_input    = build_input(inputs)
    churn_prob = model.predict_proba(X_input)[0][1]
    THRESHOLD  = 0.45
    prediction = 'Churn' if churn_prob >= THRESHOLD else 'No Churn'

    # ── Results: top row ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Churn probability', f'{churn_prob*100:.1f}%')

    with col2:
        st.metric('Prediction', prediction)

    with col3:
        if churn_prob >= 0.70:
            risk = '🔴 High risk'
        elif churn_prob >= 0.45:
            risk = '🟠 Medium risk'
        else:
            risk = '🟢 Low risk'
        st.metric('Risk level', risk)

    st.divider()

    # ── Results: probability bar ──────────────────────────────────────────────
    st.subheader('Churn probability')
    bar_color = (
        '#D85A30' if churn_prob >= 0.70 else
        '#EF9F27' if churn_prob >= 0.45 else
        '#1D9E75'
    )
    st.markdown(f'''
        <div style="background:#F1EFE8;border-radius:8px;
                    height:24px;width:100%;overflow:hidden;">
          <div style="background:{bar_color};height:100%;
                      width:{churn_prob*100:.1f}%;border-radius:8px;
                      transition:width .5s;">
          </div>
        </div>
        <p style="font-size:13px;color:#888780;margin:6px 0 0">
          {churn_prob*100:.1f}% probability of churn
          &nbsp;|&nbsp; Threshold: {THRESHOLD}
        </p>
    ''', unsafe_allow_html=True)

    st.divider()

    # ── Results: SHAP waterfall ───────────────────────────────────────────────
    st.subheader('Why this prediction? — SHAP explanation')
    st.caption(
        'Bars to the right (red) increase churn risk. '
        'Bars to the left (blue) reduce it.'
    )

    shap_vals = explainer.shap_values(X_input)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=X_input.iloc[0],
            feature_names=feature_cols
        ),
        max_display=10,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Results: top drivers table ────────────────────────────────────────────
    st.subheader('Top churn drivers for this customer')

    shap_df = pd.DataFrame({
        'Feature':    feature_cols,
        'SHAP value': shap_vals[0],
        'Direction':  ['Increases churn risk' if v > 0
                       else 'Reduces churn risk'
                       for v in shap_vals[0]]
    })
    shap_df['Abs'] = shap_df['SHAP value'].abs()
    shap_df = (shap_df
               .sort_values('Abs', ascending=False)
               .head(8)
               .drop(columns='Abs')
               .reset_index(drop=True))
    shap_df.index += 1

    st.dataframe(
        shap_df.style.applymap(
            lambda v: 'color: #D85A30' if v == 'Increases churn risk'
                      else 'color: #1D9E75',
            subset=['Direction']
        ),
        use_container_width=True
    )

    st.divider()

    # ── Results: business recommendation ─────────────────────────────────────
    st.subheader('Recommended action')

    if prediction == 'Churn':
        top_driver = shap_df.iloc[0]['Feature']

        recommendations = {
            'Contract_Two year':         '📋 Offer a discounted two-year contract upgrade',
            'charge_per_tenure':         '💰 Offer a loyalty discount — customer is new and paying a lot',
            'tenure':                    '🤝 Trigger early onboarding programme — new customer at risk',
            'InternetService_Fiber optic':'📡 Review fibre pricing or offer a bundle discount',
            'num_services':              '➕ Cross-sell one additional service to increase stickiness',
            'MonthlyCharges':            '💳 Offer a bill review or temporary discount',
            'PaymentMethod_Electronic check': '🔄 Incentivise switch to automatic payment method',
            'tenure_group_established':  '🏆 Offer a loyalty reward for reaching 2-year milestone'
        }
        action = recommendations.get(
            top_driver,
            f'🎯 Review top driver ({top_driver}) and tailor retention offer accordingly'
        )
        st.warning(f'**At-risk customer detected.**\n\n{action}')
    else:
        st.success(
            '**Low churn risk.** No immediate intervention required. '
            'Consider a standard loyalty touch point to maintain satisfaction.'
        )

else:
    # ── Default state before prediction ──────────────────────────────────────
    st.info(
        'Fill in the customer details in the sidebar and click '
        '**Predict churn risk** to see the prediction and explanation.'
    )
    st.markdown('''
    **This app will show you:**
    - Churn probability score (0–100%)
    - Risk level — low / medium / high
    - SHAP waterfall plot — exactly why the model made this prediction
    - Top 8 feature drivers for this specific customer
    - A tailored business recommendation
    ''')
