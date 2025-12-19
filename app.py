import streamlit as st
import pandas as pd
import joblib
import os
from google.cloud import storage

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

MODEL_PATH = "model.joblib"
BUCKET_NAME = "credit-risk-pred-pranav"
BLOB_NAME = "model.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        blob.download_to_filename(MODEL_PATH)
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Enter Customer Features")

# ---- INPUTS (example subset; can be expanded later) ----
LIMIT_BAL = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0.0)
age = st.number_input("Age", min_value=18)
payment_delay = st.number_input("Payment Delay Count", min_value=0)
std_paybill_ratio = st.number_input("STD Pay/Bill Ratio", value=0.0)
avg_utilization = st.number_input("Average Utilization", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL,
        "age": age,
        "payment_delay": payment_delay,
        "std_paybill_ratio": std_paybill_ratio,
        "avg_utilization": avg_utilization
    }])

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("⚠️ High Risk: Likely to Default")
    else:
        st.success("✅ Low Risk: Not Likely to Default")

