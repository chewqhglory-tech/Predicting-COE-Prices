import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Load trained model & features
# ---------------------------
model = joblib.load("coe_rf_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="COE Price Predictor", layout="centered")

st.title("COE Price Prediction Tool")
st.markdown("Predict monthly COE prices using lagged prices, quota and policy impact.")

# ---------------------------
# User Inputs
# ---------------------------

st.header("Input Parameters")

vehicle_class = st.selectbox(
    "Vehicle Class",
    ["Category A", "Category B", "Category C", "Category D", "Category E"]
)

lag1 = st.number_input(
    "COE Price (Last Month)",
    min_value=0,
    value=90000,
    step=1000
)

lag2 = st.number_input(
    "COE Price (2 Months Ago)",
    min_value=0,
    value=88000,
    step=1000
)

lag3 = st.number_input(
    "COE Price (3 Months Ago)",
    min_value=0,
    value=86000,
    step=1000
)

monthly_quota = st.number_input(
    "Monthly Quota",
    min_value=0,
    value=1200,
    step=50
)

policy_impact = st.slider(
    "Policy Impact Intensity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0 = no policy effect, 1 = maximum policy impact"
)

# ---------------------------
# Build input dataframe
# ---------------------------

input_data = {
    "Monthly COE Price_lag1": lag1,
    "Monthly COE Price_lag2": lag2,
    "Monthly COE Price_lag3": lag3,
    "Monthly Quota": monthly_quota,
    "Policy_Impact": policy_impact
}

# Vehicle class dummy variables
for col in model_features:
    if col.startswith("Vehicle_Class_"):
        input_data[col] = 0

vehicle_col = f"Vehicle_Class_{vehicle_class}"
if vehicle_col in input_data:
    input_data[vehicle_col] = 1

X_input = pd.DataFrame([input_data])

# Ensure column order matches training
X_input = X_input.reindex(columns=model_features, fill_value=0)

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict COE Price"):
    prediction = model.predict(X_input)[0]

    st.success(f"Predicted COE Price: **${prediction:,.0f}**")

    st.caption(
        "Prediction based on historical COE dynamics, quota conditions and policy regime."
    )

# ---------------------------
# Optional explanation
# ---------------------------

with st.expander("How this model works"):
    st.markdown("""
    - Uses **Random Forest Regression**
    - Learns from:
        - Price momentum (last 3 months)
        - Supply (monthly quota)
        - Policy regime intensity
    - Policy impact reflects demand shifts due to government measures
    - Trained using time-aware split (pre- vs post-policy)
    """)