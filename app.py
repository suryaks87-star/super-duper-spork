import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("bankruptcy_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Bankruptcy Prediction", page_icon="💰")

st.title("💰 Company Bankruptcy Prediction")
st.write("Enter the financial values below to predict whether the company is likely to go bankrupt.")

# ---------- INPUT FEATURES ----------
# Replace these feature names with YOUR ACTUAL feature names
features = [
    "Industrial Risk",
    "Management Risk",
    "Financial Flexibility",
    "Credibility",
    "Competitiveness",
    "Operating Risk"
]

inputs = {}

for feature in features:
    inputs[feature] = st.selectbox(
        feature,
        options=[0, 0.5, 1],
        index=2
    )

# Convert input to DataFrame
input_df = pd.DataFrame([inputs])

# Scale input
scaled_input = scaler.transform(input_df)

# Prediction
if st.button("Predict"):

    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Bankrupt")
    else:
        st.success("✅ Not Bankrupt")

    # Show probability if available
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_input)

        st.write("### Prediction Probability")
        st.write(f"Not Bankrupt : **{probability[0][0]*100:.2f}%**")
        st.write(f"Bankrupt : **{probability[0][1]*100:.2f}%**")
