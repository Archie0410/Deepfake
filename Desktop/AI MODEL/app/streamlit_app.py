# app/streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.title("💊 ADR (Adverse Drug Reaction) Risk Predictor")
st.markdown("Enter patient details to predict the likelihood of experiencing an ADR.")

# UI Inputs
age = st.slider("Age", 18, 90, 30)
gender = st.selectbox("Gender", encoders['gender'].classes_)
drug = st.selectbox("Drug", encoders['drug'].classes_)
genomics = st.selectbox("Genomic Marker", encoders['genomics'].classes_)
past_diseases = st.selectbox("Past Disease History", encoders['past_diseases'].classes_)

# Encode Inputs
gender_enc = encoders['gender'].transform([gender])[0]
drug_enc = encoders['drug'].transform([drug])[0]
genomics_enc = encoders['genomics'].transform([genomics])[0]
past_disease_enc = encoders['past_diseases'].transform([past_diseases])[0]

# Predict ADR
if st.button("🧠 Predict ADR Risk"):
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender_enc,
        "drug": drug_enc,
        "genomics": genomics_enc,
        "past_diseases": past_disease_enc
    }])
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of ADR (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of ADR (Confidence: {prob:.2f})")
