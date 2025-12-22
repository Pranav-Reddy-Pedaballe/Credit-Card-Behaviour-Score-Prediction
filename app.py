import streamlit as st
import pandas as pd
import joblib
import os
from google.cloud import storage

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

MODEL_PATH = "model.joblib"
BUCKET_NAME = "credit-risk-pred-pranav"
BLOB_NAME = "model.joblib"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)
        blob.download_to_filename(MODEL_PATH)
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess(df):
    df = df.copy()

    # 🔴 DROP Customer_ID (used only for reference)
    df.drop("Customer_ID", axis=1, inplace=True)

    # payment delay features
    df["payment_delay"] = (df[[f"pay_{i}" for i in [0,2,3,4,5,6]]] > 0).sum(axis=1)
    df["max_payment_delay"] = df[[f"pay_{i}" for i in [0,2,3,4,5,6]]].max(axis=1)

    # utilization features
    for i in range(1, 7):
        df[f"utilization_m{i}"] = df[f"Bill_amt{i}"] / df["LIMIT_BAL"]

    df["avg_utilization"] = df[[f"utilization_m{i}" for i in range(1, 7)]].mean(axis=1)
    df.drop([f"utilization_m{i}" for i in range(1, 7)], axis=1, inplace=True)

    # pay-bill ratios
    for i in range(1, 6):
        df[f"paybill_ratio_{i}"] = df[f"pay_amt{i+1}"] / (df[f"Bill_amt{i}"] + 1e-3)

    df["tot_paybill_ratio"] = (
        df[[f"pay_amt{i}" for i in range(1,7)]].sum(axis=1) /
        (df[[f"Bill_amt{i}" for i in range(1,7)]].sum(axis=1) + 1e-3)
    )

    df["std_paybill_ratio"] = df[[f"paybill_ratio_{i}" for i in range(1,6)]].std(axis=1)

    df["diff"] = (
        df[[f"Bill_amt{i}" for i in range(1,7)]].mean(axis=1)
        - df["AVG_Bill_amt"]
    )

    # drop raw bill amounts
    df.drop([f"Bill_amt{i}" for i in range(1,7)], axis=1, inplace=True)

    return df

# ---------------- UI ----------------
st.subheader("Enter Customer Features")

Customer_ID = st.number_input("Customer ID", value=0)

marriage = st.number_input("Marriage", value=1)
sex = st.number_input("Sex", value=1)
education = st.number_input("Education", value=2)
LIMIT_BAL = st.number_input("Credit Limit", min_value=0.0)
age = st.number_input("Age", min_value=18)

pay_0 = st.number_input("Pay Status Month 0", value=0)
pay_2 = st.number_input("Pay Status Month 2", value=0)
pay_3 = st.number_input("Pay Status Month 3", value=0)
pay_4 = st.number_input("Pay Status Month 4", value=0)
pay_5 = st.number_input("Pay Status Month 5", value=0)
pay_6 = st.number_input("Pay Status Month 6", value=0)

Bill_amt1 = st.number_input("Bill Amount 1", value=0.0)
Bill_amt2 = st.number_input("Bill Amount 2", value=0.0)
Bill_amt3 = st.number_input("Bill Amount 3", value=0.0)
Bill_amt4 = st.number_input("Bill Amount 4", value=0.0)
Bill_amt5 = st.number_input("Bill Amount 5", value=0.0)
Bill_amt6 = st.number_input("Bill Amount 6", value=0.0)

pay_amt1 = st.number_input("Pay Amount 1", value=0.0)
pay_amt2 = st.number_input("Pay Amount 2", value=0.0)
pay_amt3 = st.number_input("Pay Amount 3", value=0.0)
pay_amt4 = st.number_input("Pay Amount 4", value=0.0)
pay_amt5 = st.number_input("Pay Amount 5", value=0.0)
pay_amt6 = st.number_input("Pay Amount 6", value=0.0)

AVG_Bill_amt = st.number_input("Average Bill Amount", value=0.0)

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    raw_df = pd.DataFrame([{
        "Customer_ID": Customer_ID,
        "marriage": marriage,
        "sex": sex,
        "education": education,
        "LIMIT_BAL": LIMIT_BAL,
        "age": age,
        "pay_0": pay_0,
        "pay_2": pay_2,
        "pay_3": pay_3,
        "pay_4": pay_4,
        "pay_5": pay_5,
        "pay_6": pay_6,
        "Bill_amt1": Bill_amt1,
        "Bill_amt2": Bill_amt2,
        "Bill_amt3": Bill_amt3,
        "Bill_amt4": Bill_amt4,
        "Bill_amt5": Bill_amt5,
        "Bill_amt6": Bill_amt6,
        "pay_amt1": pay_amt1,
        "pay_amt2": pay_amt2,
        "pay_amt3": pay_amt3,
        "pay_amt4": pay_amt4,
        "pay_amt5": pay_amt5,
        "pay_amt6": pay_amt6,
        "AVG_Bill_amt": AVG_Bill_amt
    }])

    processed_df = preprocess(raw_df)

    # enforce training feature order
    processed_df = processed_df[model.feature_names_in_]

    pred = model.predict(processed_df)[0]

    if pred == 1:
        st.error(f"⚠️ Customer {Customer_ID}: Likely to Default")
    else:
        st.success(f"✅ Customer {Customer_ID}: Not Likely to Default")
