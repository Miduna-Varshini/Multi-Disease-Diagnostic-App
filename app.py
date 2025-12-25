import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction (13 Features)")

# ================= SAFE MODEL LOADER =================
@st.cache_resource
def load_model():
    with open("models/heart_model.pkl", "rb") as f:
        data = pickle.load(f)
    # Handle both tuple and dict formats
    if isinstance(data, tuple):
        model, scaler = data
        features = [
            "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
            "FBS over 120", "EKG results", "Max HR", "Exercise angina",
            "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"
        ]
    else:
        model = data["model"]
        scaler = data["scaler"]
        features = data["features"]
    return model, scaler, features

model, scaler, FEATURES = load_model()

# ================= INPUTS =================
st.subheader("Enter Patient Details")

age = st.number_input("Age", 0, 120, value=52)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0–3)", 0, 3, value=0)
trestbps = st.number_input("Resting Blood Pressure (BP)", 80, 200, value=120)
chol = st.number_input("Cholesterol", 100, 600, value=240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.number_input("Resting ECG Results (0–2)", 0, 2, value=1)
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 250, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, value=1.2)
slope = st.number_input("Slope of ST Segment (0–2)", 0, 2, value=1)
ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, value=0)
thal = st.number_input("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", 1, 3, value=2)

# ================= PREDICTION =================
if st.button("🔍 Predict Heart Disease"):
    try:
        # Quick rule-based alert for obvious risk
        if chol > 300 or trestbps > 160 or thalach < 100:
            st.error("⚠️ Possible Heart Disease Detected (Rule-Based Alert)")
        else:
            # Prepare input for model
            X_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]])
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]

            if prediction == 1:
                st.error("⚠️ Heart Disease Detected")
            else:
                st.success("✅ No Heart Disease Detected")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))

st.markdown("---")
st.markdown("Made with ❤️ by your ML buddy")
