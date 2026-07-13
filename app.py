import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load Model and Scaler
# ===============================
model = joblib.load("bankruptcy_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Company Bankruptcy Prediction")
st.write("Select the values for each feature and click **Predict**.")

st.markdown("---")

# ===============================
# User Inputs
# ===============================

industrial_risk = st.selectbox(
    "Industrial Risk",
    options=[0, 0.5, 1],
    index=0
)

management_risk = st.selectbox(
    "Management Risk",
    options=[0, 0.5, 1],
    index=0
)

financial_flexibility = st.selectbox(
    "Financial Flexibility",
    options=[0, 0.5, 1],
    index=2
)

credibility = st.selectbox(
    "Credibility",
    options=[0, 0.5, 1],
    index=2
)

competitiveness = st.selectbox(
    "Competitiveness",
    options=[0, 0.5, 1],
    index=2
)

operating_risk = st.selectbox(
    "Operating Risk",
    options=[0, 0.5, 1],
    index=0
)

# ===============================
# Predict Button
# ===============================

if st.button("Predict"):

    # Create DataFrame with EXACT feature names
    input_df = pd.DataFrame(
        [[
            industrial_risk,
            management_risk,
            financial_flexibility,
            credibility,
            competitiveness,
            operating_risk
        ]],
        columns=[
            "industrial_risk",
            "management_risk",
            "financial_flexibility",
            "credibility",
            "competitiveness",
            "operating_risk"
        ]
    )

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Bankrupt")
    else:
        st.success("✅ Not Bankrupt")

    # Display probabilities
    if hasattr(model, "predict_proba"):
        st.subheader("Prediction Probability")

        st.write(f"**Not Bankrupt:** {probability[0] * 100:.2f}%")
        st.write(f"**Bankrupt:** {probability[1] * 100:.2f}%")
