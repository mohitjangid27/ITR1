"""Heart Disease Predictor - Streamlit app.
Run:  streamlit run app.py      (train first: python train_model.py)
"""
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(160deg,#0e1117 0%,#16313d 100%);}
h1.title {text-align:center; font-size:3.2rem; color:white; margin-top:1rem;}
p.sub {text-align:center; font-size:1.3rem; color:#c9d4da; margin-bottom:2.5rem;}
div.stButton > button {background:#ff4b4b; color:white; border:none; padding:.7rem 1.4rem;
    font-size:1.05rem; border-radius:.6rem;}
div.stButton > button:hover {background:#e03a3a; color:white;}
</style>
<h1 class="title">Heart Disease Predictor</h1>
<p class="sub">AI Powered Medical Risk Detection System</p>
""", unsafe_allow_html=True)

if not os.path.exists("model.joblib"):
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()
model = joblib.load("model.joblib")

# label -> UCI numeric code
CP = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
ECG = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
YN = {"No": 0, "Yes": 1}

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", 1, 120, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", list(CP), index=2)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50.0, 250.0, 120.0)
with c2:
    chol = st.number_input("Cholesterol (mg/dl)", 100.0, 600.0, 200.0)
    fbs = st.selectbox("Fasting Blood Sugar >120", list(YN))
    restecg = st.selectbox("Rest ECG", list(ECG))
    thalach = st.number_input("Max Heart Rate", 50.0, 250.0, 150.0)
with c3:
    exang = st.selectbox("Exercise Induced Angina", list(YN))
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, step=0.1)
    slope = st.selectbox("Slope", list(SLOPE))
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", list(THAL))

st.write("")
if st.button("Predict Heart Disease"):
    row = pd.DataFrame([[age, 1 if gender == "Male" else 0, CP[cp], trestbps, chol,
                         YN[fbs], ECG[restecg], thalach, YN[exang], oldpeak,
                         SLOPE[slope], ca, THAL[thal]]],
                       columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                                "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
    p = float(model.predict_proba(row)[0, 1])
    level = "LOW" if p < 0.30 else "MODERATE" if p < 0.60 else "HIGH"
    st.subheader(f"Estimated heart disease risk: {p:.1%}")
    st.progress(min(p, 1.0))
    msg = {"LOW": ("success", "Low risk. Keep up a healthy lifestyle and regular check-ups."),
           "MODERATE": ("warning", "Moderate risk. Consider a check-up with a doctor."),
           "HIGH": ("error", "High risk. Please consult a cardiologist soon.")}[level]
    getattr(st, msg[0])(f"**{level} RISK** - {msg[1]}")
    st.caption("Educational project only - not a medical diagnosis.")
