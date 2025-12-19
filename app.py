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
Customer_ID= st.number_input("Customer_ID", value=0.0)
marriage = st.number_input("marriage", value=0.0)
sex = st.number_input("sex", value=0.0)
education = st.number_input("education", value=0.0)
LIMIT_BAL = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0.0)
age = st.number_input("Age", min_value=18)
pay_0 = st.number_input("pay_0", value=0.0)
pay_2 = st.number_input("pay_2", value=0.0)
pay_3 = st.number_input("pay_3", value=0.0)
pay_4 = st.number_input("pay_4", value=0.0)
pay_5 = st.number_input("pay_5", value=0.0)
pay_6 = st.number_input("pay_6", value=0.0)
Bill_amt1 = st.number_input("Bill_amt1", value=0.0)
Bill_amt2 = st.number_input("Bill_amt2", value=0.0)
Bill_amt3 = st.number_input("Bill_amt3", value=0.0)
Bill_amt4 = st.number_input("Bill_amt4", value=0.0)
Bill_amt5 = st.number_input("Bill_amt5", value=0.0)
Bill_amt6 = st.number_input("Bill_amt6", value=0.0)
pay_amt1 = st.number_input("pay_amt1", value=0.0)
pay_amt2 = st.number_input("pay_amt2", value=0.0)
pay_amt3 = st.number_input("pay_amt3", value=0.0)
pay_amt4 = st.number_input("pay_amt4", value=0.0)
pay_amt5 = st.number_input("pay_amt5", value=0.0)
pay_amt6 = st.number_input("pay_amt6", value=0.0)
AVG_Bill_amt = st.number_input("AVG_Bill_amt", value=0.0)
PAY_TO_BILL_ratio = st.number_input("PAY_TO_BILL_ratio", value=0.0)


if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Customer_ID"= Customer_ID
        "marriage" = marriage
        "sex" = sex
        "education" = education
        "LIMIT_BAL" = LIMIT_BAL
        "age" = st.number_input("Age", min_value=18)
        "pay_0" = st.number_input("pay_0", value=0.0)
        "pay_2" = st.number_input("pay_2", value=0.0)
        "pay_3" = st.number_input("pay_3", value=0.0)
        "pay_4" = st.number_input("pay_4", value=0.0)
        "pay_5" = st.number_input("pay_5", value=0.0)
        "pay_6" = st.number_input("pay_6", value=0.0)
        "Bill_amt1" = st.number_input("Bill_amt1", value=0.0)
        "Bill_amt2" = st.number_input("Bill_amt2", value=0.0)
        "Bill_amt3" = st.number_input("Bill_amt3", value=0.0)
        "Bill_amt4" = st.number_input("Bill_amt4", value=0.0)
        "Bill_amt5" = st.number_input("Bill_amt5", value=0.0)
        "Bill_amt6" = st.number_input("Bill_amt6", value=0.0)
        "pay_amt1" = st.number_input("pay_amt1", value=0.0)
        "pay_amt2" = st.number_input("pay_amt2", value=0.0)
        "pay_amt3" = st.number_input("pay_amt3", value=0.0)
        "pay_amt4" = st.number_input("pay_amt4", value=0.0)
        "pay_amt5" = st.number_input("pay_amt5", value=0.0)
        "pay_amt6" = st.number_input("pay_amt6", value=0.0)
        "AVG_Bill_amt" = st.number_input("AVG_Bill_amt", value=0.0)
        "PAY_TO_BILL_ratio" = st.number_input("PAY_TO_BILL_ratio", value=0.0)
    }])

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("⚠️ High Risk: Likely to Default")
    else:
        st.success("✅ Low Risk: Not Likely to Default")


