import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("bankruptcy_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Company Bankruptcy Prediction")

st.write("Select the values below and click Predict.")

st.markdown("---")

# -----------------------------
# Input Fields
# -----------------------------

industrial_risk = st.selectbox(
    "Industrial Risk",
    [0, 0.5, 1],
    index=0
)

management_risk = st.selectbox(
    "Management Risk",
    [0, 0.5, 1],
    index=0
)

financial_flexibility = st.selectbox(
    "Financial Flexibility",
    [0, 0.5, 1],
    index=2
)

credibility = st.selectbox(
    "Credibility",
    [0, 0.5, 1],
    index=2
)

competitiveness = st.selectbox(
    "Competitiveness",
    [0, 0.5, 1],
    index=2
)

operating_risk = st.selectbox(
    "Operating Risk",
    [0, 0.5, 1],
    index=0
)

# -----------------------------
# Create DataFrame
# -----------------------------

input_data = pd.DataFrame({
    "industrial_risk": [industrial_risk],
    "management_risk": [management_risk],
    "financial_flexibility": [financial_flexibility],
    "credibility": [credibility],
    "competitiveness": [competitiveness],
    "operating_risk": [operating_risk]
})

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Prediction: Bankrupt")
    else:
        st.success("✅ Prediction: Not Bankrupt")

    st.subheader("Prediction Probability")

    st.write(f"Not Bankrupt : **{probability[0]*100:.2f}%**")

    st.write(f"Bankrupt : **{probability[1]*100:.2f}%**")
