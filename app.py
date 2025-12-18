import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model_and_features():
    model = joblib.load("coe_rf_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, features = load_model_and_features()

st.title("🚗 COE Price Predictor")

vehicle_class = st.selectbox(
    "Vehicle Class",
    ["Category A", "Category B", "Category C", "Category D", "Category E"]
)

lag1 = st.number_input("COE Price (Last Month)", min_value=0)
lag2 = st.number_input("COE Price (2 Months Ago)", min_value=0)
lag3 = st.number_input("COE Price (3 Months Ago)", min_value=0)

monthly_quota = st.number_input("Monthly Quota", min_value=0)
bids_received = st.number_input("Monthly Bids Received", min_value=0)
bids_successful = st.number_input("Monthly Bids Successful", min_value=0)
dereg = st.number_input("Deregistration Value", min_value=0)
new_reg = st.number_input("New Registration Value", min_value=0)
sora_3m = st.slider("SORA 3M (%)", 0.0, 6.0, 2.5)
cpi = st.slider("CPI Private Transport", 90.0, 130.0, 110.0)

input_df = pd.DataFrame({
    "Monthly Quota": [monthly_quota],
    "Monthly Bids Received": [bids_received],
    "Monthly Bids Successful": [bids_successful],
    "Deregistration_Value": [dereg],
    "New_registration_Value": [new_reg],
    "SORA_3M": [sora_3m],
    "CPI Private Transport": [cpi],
    "Monthly COE Price_lag1": [lag1],
    "Monthly COE Price_lag2": [lag2],
    "Monthly COE Price_lag3": [lag3],
    "Vehicle_Class_Category B": [1 if vehicle_class == "Category B" else 0],
    "Vehicle_Class_Category C": [1 if vehicle_class == "Category C" else 0],
    "Vehicle_Class_Category D": [1 if vehicle_class == "Category D" else 0],
    "Vehicle_Class_Category E": [1 if vehicle_class == "Category E" else 0],
})

input_df = input_df[features]

if st.button("Predict COE Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted COE Price: ${prediction:,.0f}")
