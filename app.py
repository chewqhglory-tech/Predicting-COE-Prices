import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model & features
# -----------------------------
model = joblib.load("coe_rf_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="COE Price Predictor", layout="centered")

st.title("🚗 COE Price Prediction Tool")
st.write(
    """
    Predict monthly COE prices using:
    - Last 3 months COE prices  
    - SORA (3M)  
    - CPI – Private Transport  
    - Government policy impact
    """
)

# -----------------------------
# User Inputs
# -----------------------------
st.header("Input Parameters")

lag1 = st.number_input(
    "COE Price (Last Month)",
    min_value=0.0,
    value=80000.0,
    step=1000.0
)

lag2 = st.number_input(
    "COE Price (2 Months Ago)",
    min_value=0.0,
    value=78000.0,
    step=1000.0
)

lag3 = st.number_input(
    "COE Price (3 Months Ago)",
    min_value=0.0,
    value=76000.0,
    step=1000.0
)

sora_3m = st.number_input(
    "SORA 3M (%)",
    min_value=0.0,
    value=3.5,
    step=0.1
)

cpi_private = st.number_input(
    "CPI – Private Transport",
    min_value=0.0,
    value=110.0,
    step=0.5
)

policy_impact = st.slider(
    "Policy Impact Intensity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0 = no policy impact, 1 = strong policy effect"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict COE Price"):
    
    # Create input dataframe
    input_data = pd.DataFrame([{
        "Monthly COE Price_lag1": lag1,
        "Monthly COE Price_lag2": lag2,
        "Monthly COE Price_lag3": lag3,
        "SORA_3M": sora_3m,
        "CPI Private Transport": cpi_private,
        "Policy_Impact": policy_impact
    }])

    # Ensure all expected model features exist
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0

    # Align column order
    input_data = input_data[model_features]

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"📈 Predicted COE Price: **${prediction:,.0f}**")

    st.caption(
        "Prediction is based on historical relationships learned by the Random Forest model."
    )