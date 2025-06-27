import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load trained model and encoders
model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.set_page_config(page_title="ADR Severity Predictor", page_icon="💊", layout="centered")
st.title("💊 ADR Severity Risk Predictor")
st.markdown("Fill in patient details to estimate the severity of an Adverse Drug Reaction (ADR).")

# UI Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 60)
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    drug = st.selectbox("Drug", encoders["drug"].classes_)
    genomics = st.selectbox("Genomic Marker", encoders["genomics"].classes_)
    past_diseases = st.selectbox("Past Disease History", encoders["past_diseases"].classes_)
    reason_for_drug = st.selectbox("Reason for Drug", encoders["reason_for_drug"].classes_)
    drug_quantity = st.slider("Number of Drugs Prescribed", 1, 10, 2)

with col2:
    allergies = st.selectbox("Allergies", encoders["allergies"].classes_)
    addiction = st.selectbox("Addiction", encoders["addiction"].classes_)
    ayurvedic = st.selectbox("Ayurvedic Medicine Use", encoders["ayurvedic_medicine"].classes_)
    hereditary = st.selectbox("Hereditary Disease", encoders["hereditary_disease"].classes_)
    drug_duration = st.slider("Drug Duration (days)", 1, 180, 30)
    age_group = st.selectbox("Age Group", encoders["age_group"].classes_)

# Compute polypharmacy_flag dynamically
polypharmacy_flag = 1 if drug_quantity > 3 else 0

# Encode categorical values
def encode(col, val):
    return encoders[col].transform([val])[0]

input_data = {
    "age": age,
    "gender": encode("gender", gender),
    "drug": encode("drug", drug),
    "genomics": encode("genomics", genomics),
    "past_diseases": encode("past_diseases", past_diseases),
    "reason_for_drug": encode("reason_for_drug", reason_for_drug),
    "drug_quantity": drug_quantity,
    "allergies": encode("allergies", allergies),
    "addiction": encode("addiction", addiction),
    "ayurvedic_medicine": encode("ayurvedic_medicine", ayurvedic),
    "hereditary_disease": encode("hereditary_disease", hereditary),
    "drug_duration": drug_duration,
    "age_group": encode("age_group", age_group),
    "polypharmacy_flag": polypharmacy_flag
}

input_df = pd.DataFrame([input_data])

# Ensure order matches training
expected_order = [
    "age", "gender", "drug", "genomics", "past_diseases", "reason_for_drug",
    "drug_quantity", "allergies", "addiction", "ayurvedic_medicine",
    "hereditary_disease", "drug_duration", "age_group", "polypharmacy_flag"
]
input_df = input_df[expected_order]

# Predict ADR severity
if st.button("🧠 Predict ADR Severity"):
    prediction_encoded = model.predict(input_df)[0]
    predicted_label = encoders["adr_severity"].inverse_transform([prediction_encoded])[0]
    prob = max(model.predict_proba(input_df)[0])

    st.subheader("🔍 Prediction Result")
    color_map = {"Low": "green", "Medium": "orange", "High": "red"}
    st.markdown(f"### 🔬 **Predicted ADR Severity:** `{predicted_label}` (Confidence: `{prob:.2f}`)")

    # Gauge chart
    severity_levels = ["Low", "Medium", "High"]
    value_map = {level: i * 50 for i, level in enumerate(severity_levels)}
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_map[predicted_label],
        title={"text": "ADR Severity Level"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color_map[predicted_label]},
            "steps": [
                {"range": [0, 50], "color": "#d4edda"},
                {"range": [50, 100], "color": "#fff3cd"},
                {"range": [100, 150], "color": "#f8d7da"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
