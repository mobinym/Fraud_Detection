import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and expected columns
model_path = '/Users/mobinyaghoobi/Downloads/تمرین 6/Exercises 4/notebooks/best_gradient_boosting_model.pkl'
scaler_path = '/Users/mobinyaghoobi/Downloads/تمرین 6/Exercises 4/notebooks/scaler.pkl'
columns_path = '/Users/mobinyaghoobi/Downloads/تمرین 6/Exercises 4/notebooks/expected_columns.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_columns = joblib.load(columns_path)

# Streamlit UI
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.title("Fraud Detection System")
st.sidebar.header("Input Features")

# Collect input data
input_data = {
    "Transaction_Amount": st.sidebar.number_input("Transaction Amount", min_value=0.0, step=0.01),
    "Account_Balance": st.sidebar.number_input("Account Balance", min_value=0.0, step=0.01),
    "Card_Age": st.sidebar.number_input("Card Age", min_value=0, step=1),
    "Transaction_Distance": st.sidebar.number_input("Transaction Distance", min_value=0.0, step=0.01),
    "Risk_Score": st.sidebar.number_input("Risk Score", min_value=0.0, max_value=1.0, step=0.01),
    "Is_Weekend": 1 if st.sidebar.selectbox("Is Weekend?", ["Yes", "No"]) == "Yes" else 0,
    "Year": st.sidebar.slider("Year", min_value=2000, max_value=2025, step=1),
    "Month": st.sidebar.slider("Month", min_value=1, max_value=12, step=1),
    "Day": st.sidebar.slider("Day", min_value=1, max_value=31, step=1),
    "Day_of_Week": st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    "Hour": st.sidebar.slider("Hour of Transaction", min_value=0, max_value=23, step=1),
    "Transaction_Ratio": st.sidebar.number_input("Transaction Ratio", min_value=0.0, step=0.01),
    "Failed_Transaction_Rate": st.sidebar.number_input("Failed Transaction Rate", min_value=0.0, step=0.01),
    "Transaction_Type_Bank Transfer": 1 if st.sidebar.selectbox("Transaction Type", ["Bank Transfer", "Online", "POS", "Other"]) == "Bank Transfer" else 0,
    "Transaction_Type_Online": 1 if st.sidebar.selectbox("Transaction Type", ["Bank Transfer", "Online", "POS", "Other"]) == "Online" else 0,
    "Transaction_Type_POS": 1 if st.sidebar.selectbox("Transaction Type", ["Bank Transfer", "Online", "POS", "Other"]) == "POS" else 0,
    "Device_Type_Mobile": 1 if st.sidebar.selectbox("Device Type", ["Mobile", "Tablet", "Other"]) == "Mobile" else 0,
    "Device_Type_Tablet": 1 if st.sidebar.selectbox("Device Type", ["Mobile", "Tablet", "Other"]) == "Tablet" else 0,
    "Location_Mumbai": 1 if st.sidebar.selectbox("Location", ["Mumbai", "New York", "Sydney", "Tokyo"]) == "Mumbai" else 0,
    "Location_New York": 1 if st.sidebar.selectbox("Location", ["Mumbai", "New York", "Sydney", "Tokyo"]) == "New York" else 0,
    "Location_Sydney": 1 if st.sidebar.selectbox("Location", ["Mumbai", "New York", "Sydney", "Tokyo"]) == "Sydney" else 0,
    "Location_Tokyo": 1 if st.sidebar.selectbox("Location", ["Mumbai", "New York", "Sydney", "Tokyo"]) == "Tokyo" else 0,
    "Merchant_Category_Electronics": 1 if st.sidebar.selectbox("Merchant Category", ["Electronics", "Groceries", "Restaurants", "Travel"]) == "Electronics" else 0,
    "Merchant_Category_Groceries": 1 if st.sidebar.selectbox("Merchant Category", ["Electronics", "Groceries", "Restaurants", "Travel"]) == "Groceries" else 0,
    "Merchant_Category_Restaurants": 1 if st.sidebar.selectbox("Merchant Category", ["Electronics", "Groceries", "Restaurants", "Travel"]) == "Restaurants" else 0,
    "Merchant_Category_Travel": 1 if st.sidebar.selectbox("Merchant Category", ["Electronics", "Groceries", "Restaurants", "Travel"]) == "Travel" else 0,
    "Card_Type_Discover": 1 if st.sidebar.selectbox("Card Type", ["Discover", "Mastercard", "Visa"]) == "Discover" else 0,
    "Card_Type_Mastercard": 1 if st.sidebar.selectbox("Card Type", ["Discover", "Mastercard", "Visa"]) == "Mastercard" else 0,
    "Card_Type_Visa": 1 if st.sidebar.selectbox("Card Type", ["Discover", "Mastercard", "Visa"]) == "Visa" else 0,
    "Authentication_Method_OTP": 1 if st.sidebar.selectbox("Authentication Method", ["OTP", "PIN", "Password"]) == "OTP" else 0,
    "Authentication_Method_PIN": 1 if st.sidebar.selectbox("Authentication Method", ["OTP", "PIN", "Password"]) == "PIN" else 0,
    "Authentication_Method_Password": 1 if st.sidebar.selectbox("Authentication Method", ["OTP", "PIN", "Password"]) == "Password" else 0
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure all columns are present
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with default value

# Reorder columns to match model expectations
input_df = input_df[expected_columns]

if st.sidebar.button("Predict"):
    try:
        # Scale data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Display result
        st.write(f"Prediction: {'Fraudulent' if prediction[0] == 1 else 'Non-Fraudulent'}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")