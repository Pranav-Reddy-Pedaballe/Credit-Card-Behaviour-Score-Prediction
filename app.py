import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

st.title("Credit Card Behaviour Score Prediction")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    st.write(preds)
