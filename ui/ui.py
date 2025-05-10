import streamlit as st
import pandas as pd
import numpy as np
import joblib

model_path = '/Users/mobinyaghoobi/Downloads/تمرین 6/Exercises 4/notebooks/best_gradient_boosting_model.pkl'
scaler_path = '/Users/mobinyaghoobi/Downloads/تمرین 6/Exercises 4/notebooks/scaler.pkl'
columns_path = '/Users/mobinyaghoobi/Downloads/تمرین 6/Exercises 4/notebooks/expected_columns.pkl'

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    expected_columns = joblib.load(columns_path)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .title {
        font-size: 48px;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        color: #555555;
        text-align: center;
        margin-bottom: 40px;
    }
    .footer {
        font-size: 14px;
        color: #999999;
        text-align: center;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict potential fraudulent transactions with high accuracy</div>', unsafe_allow_html=True)

with st.sidebar.form("input_form"):
    st.header("Enter Transaction Details")
    st.write("**Basic Transaction Details**")
    transaction_amount = st.number_input("Transaction Amount (in USD)", min_value=0.0, step=0.01)
    account_balance = st.number_input("Account Balance (in USD)", min_value=0.0, step=0.01)
    card_age = st.number_input("Card Age (in months)", min_value=0, step=1)
    transaction_distance = st.number_input("Transaction Distance (in miles)", min_value=0.0, step=0.01)
    risk_score = st.number_input("Risk Score (0-1)", min_value=0.0, max_value=1.0, step=0.01)
    is_weekend = st.selectbox("Is Weekend?", ["Yes", "No"])
    
    st.write("**Transaction Timing**")
    year = st.slider("Year", min_value=2000, max_value=2025, step=1)
    month = st.slider("Month", min_value=1, max_value=12, step=1)
    day = st.slider("Day", min_value=1, max_value=31, step=1)
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    hour = st.slider("Hour of Transaction", min_value=0, max_value=23, step=1)

    st.write("**Transaction Details**")
    transaction_ratio = st.number_input("Transaction Ratio", min_value=0.0, step=0.01)
    failed_transaction_rate = st.number_input("Failed Transaction Rate (%)", min_value=0.0, max_value=100.0, step=0.01)

    transaction_type = st.selectbox("Transaction Type", ["Bank Transfer", "Online", "POS", "Other"])
    device_type = st.selectbox("Device Type", ["Mobile", "Tablet", "Other"])
    location = st.selectbox("Location", ["Mumbai", "New York", "Sydney", "Tokyo"])
    merchant_category = st.selectbox("Merchant Category", ["Electronics", "Groceries", "Restaurants", "Travel"])
    card_type = st.selectbox("Card Type", ["Discover", "Mastercard", "Visa"])
    authentication_method = st.selectbox("Authentication Method", ["OTP", "PIN", "Password"])

    submit_button = st.form_submit_button("Predict")

if submit_button:
    try:
        input_data = {
            "Transaction_Amount": transaction_amount,
            "Account_Balance": account_balance,
            "Card_Age": card_age,
            "Transaction_Distance": transaction_distance,
            "Risk_Score": risk_score,
            "Is_Weekend": 1 if is_weekend == "Yes" else 0,
            "Year": year,
            "Month": month,
            "Day": day,
            "Day_of_Week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week),
            "Hour": hour,
            "Transaction_Ratio": transaction_ratio,
            "Failed_Transaction_Rate": failed_transaction_rate / 100,  
            "Transaction_Type_Bank Transfer": 1 if transaction_type == "Bank Transfer" else 0,
            "Transaction_Type_Online": 1 if transaction_type == "Online" else 0,
            "Transaction_Type_POS": 1 if transaction_type == "POS" else 0,
            "Device_Type_Mobile": 1 if device_type == "Mobile" else 0,
            "Device_Type_Tablet": 1 if device_type == "Tablet" else 0,
            "Location_Mumbai": 1 if location == "Mumbai" else 0,
            "Location_New York": 1 if location == "New York" else 0,
            "Location_Sydney": 1 if location == "Sydney" else 0,
            "Location_Tokyo": 1 if location == "Tokyo" else 0,
            "Merchant_Category_Electronics": 1 if merchant_category == "Electronics" else 0,
            "Merchant_Category_Groceries": 1 if merchant_category == "Groceries" else 0,
            "Merchant_Category_Restaurants": 1 if merchant_category == "Restaurants" else 0,
            "Merchant_Category_Travel": 1 if merchant_category == "Travel" else 0,
            "Card_Type_Discover": 1 if card_type == "Discover" else 0,
            "Card_Type_Mastercard": 1 if card_type == "Mastercard" else 0,
            "Card_Type_Visa": 1 if card_type == "Visa" else 0,
            "Authentication_Method_OTP": 1 if authentication_method == "OTP" else 0,
            "Authentication_Method_PIN": 1 if authentication_method == "PIN" else 0,
            "Authentication_Method_Password": 1 if authentication_method == "Password" else 0
        }

        input_df = pd.DataFrame([input_data])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0  

        input_df = input_df[expected_columns]

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.metric(label="Transaction Status", value="Fraudulent", delta="High Risk")
            st.error("⚠️ This transaction is likely fraudulent. Please investigate further.")
        else:
            st.metric(label="Transaction Status", value="Non-Fraudulent", delta="Low Risk")
            st.success("✅ This transaction appears to be legitimate.")

        st.markdown("### Input Summary")
        st.table(input_df)

    except Exception as e:
        st.error(f"Unexpected error: {e}")

st.markdown('<div class="footer">Designed by Mobin YM</div>', unsafe_allow_html=True)