
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("coe_rf_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="COE Price Predictor", layout="centered")

st.title("🚗 COE Price Predictor")
st.write("Predict monthly COE prices using past prices and economic conditions.")

vehicle_class = st.selectbox(
    "Vehicle Class",
    ["Category A", "Category B", "Category C", "Category D"]
)

lag1 = st.number_input("COE Price Last Month ($)", min_value=0, value=80000, step=1000)
lag2 = st.number_input("COE Price 2 Months Ago ($)", min_value=0, value=78000, step=1000)
lag3 = st.number_input("COE Price 3 Months Ago ($)", min_value=0, value=76000, step=1000)

sora = st.slider("SORA (%)", 0.0, 6.0, 3.0, 0.1)
quota = st.number_input("Vehicle Quota", min_value=0, value=3000, step=100)

input_df = pd.DataFrame(0, index=[0], columns=features)
input_df["Monthly COE Price_lag1"] = lag1
input_df["Monthly COE Price_lag2"] = lag2
input_df["Monthly COE Price_lag3"] = lag3
input_df["SORA"] = sora
input_df["Vehicle_Quota"] = quota

if vehicle_class != "Category A":
    input_df[f"Vehicle_Class_{vehicle_class}"] = 1

if st.button("Predict COE Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted COE Price: ${prediction:,.0f}")
