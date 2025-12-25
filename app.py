import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Multi-Disease Diagnostic System",
    layout="centered"
)

st.title("🩺 AI Multi-Disease Diagnostic Portal")

# ================= SAFE MODEL LOADER (TABULAR) =================
@st.cache_resource
def load_tabular_model(path, default_features):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple):
        model, scaler = data
        features = default_features
    else:
        model = data["model"]
        scaler = data["scaler"]
        features = data["features"]

    return model, scaler, features

# ================= LOAD BRAIN MODEL =================
@st.cache_resource
def load_brain_model():
    FILE_ID = "1r7Kmf14ZGKQK3GSTk3nxPxfAyGpg2m_b"
    URL = f"https://drive.google.com/uc?id={FILE_ID}"

    model_path = "brain_tumor_model.h5"

    if not os.path.exists(model_path):
        response = requests.get(URL)
        with open(model_path, "wb") as f:
            f.write(response.content)

    model = load_model(model_path)
    return model

# ================= DISEASE SELECTION =================
disease = st.selectbox(
    "Choose a disease to predict:",
    ["Heart", "Kidney", "Liver", "Diabetes", "Brain Tumor"]
)

# ================= HEART =================
if disease == "Heart":
    model, scaler, _ = load_tabular_model(
        "models/heart_model.pkl",
        ["Age","Sex","CP","BP","Chol","FBS","ECG","HR","Angina","Oldpeak","Slope","CA","Thal"]
    )

    st.subheader("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 0, 120, 52)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
    cp = st.number_input("Chest Pain (0–3)", 0, 3)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 240)
    fbs = st.selectbox("FBS > 120", [0,1])
    ecg = st.number_input("ECG (0–2)", 0, 2)
    hr = st.number_input("Max Heart Rate", 60, 250, 150)
    angina = st.selectbox("Exercise Angina", [0,1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.2)
    slope = st.number_input("Slope (0–2)", 0, 2)
    ca = st.number_input("CA (0–3)", 0, 3)
    thal = st.number_input("Thal (1–3)", 1, 3)

    if st.button("🔍 Predict Heart Disease"):
        X = np.array([[age,sex,cp,bp,chol,fbs,ecg,hr,angina,oldpeak,slope,ca,thal]])
        X = scaler.transform(X)
        pred = model.predict(X)[0]
        st.error("⚠️ Heart Disease Detected" if pred == 1 else "✅ No Heart Disease")

# ================= KIDNEY =================
elif disease == "Kidney":
    model, scaler, _ = load_tabular_model("models/kidney_10f_model.pkl", [])

    st.subheader("🧪 Kidney Disease Prediction")

    age = st.number_input("Age", 0, 120, 45)
    bp = st.number_input("Blood Pressure", 0, 200, 80)
    sg = st.number_input("Specific Gravity", 1.0, 1.05, 1.020)
    al = st.number_input("Albumin", 0, 5)
    su = st.number_input("Sugar", 0, 5)
    bgr = st.number_input("Blood Glucose", 0, 500, 110)
    bu = st.number_input("Blood Urea", 0, 200, 25)
    sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.0)
    hemo = st.number_input("Hemoglobin", 0.0, 20.0, 15.2)
    pcv = st.number_input("PCV", 0, 60, 44)

    if st.button("🔍 Predict Kidney Disease"):
        X = np.array([[age,bp,sg,al,su,bgr,bu,sc,hemo,pcv]])
        X = scaler.transform(X)
        pred = model.predict(X)[0]
        st.error("⚠️ Kidney Disease Detected" if pred == 1 else "✅ No Kidney Disease")

# ================= LIVER =================
elif disease == "Liver":
    model, scaler, _ = load_tabular_model("models/liver_model.pkl", [])

    st.subheader("🧬 Liver Disease Prediction")

    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male","Female"])
    g = 1 if gender=="Male" else 0
    tb = st.number_input("Total Bilirubin", 0.0, 10.0, 1.3)
    db = st.number_input("Direct Bilirubin", 0.0, 5.0, 0.4)
    alk = st.number_input("Alkaline Phosphotase", 50, 2000, 210)
    alt = st.number_input("ALT", 1, 2000, 35)
    ast = st.number_input("AST", 1, 2000, 40)
    tp = st.number_input("Total Proteins", 1.0, 10.0, 6.8)
    alb = st.number_input("Albumin", 1.0, 6.0, 3.1)
    ag = st.number_input("A/G Ratio", 0.0, 3.0, 0.9)

    if st.button("🔍 Predict Liver Disease"):
        X = np.array([[age,g,tb,db,alk,alt,ast,tp,alb,ag]])
        X = scaler.transform(X)
        pred = model.predict(X)[0]
        st.error("⚠️ Liver Disease Detected" if pred == 1 else "✅ No Liver Disease")

# ================= DIABETES =================
elif disease == "Diabetes":
    model, scaler, _ = load_tabular_model("models/diabetes_model.pkl", [])

    st.subheader("🩸 Diabetes Prediction")

    preg = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 85)
    bmi = st.number_input("BMI", 0.0, 70.0, 28.5)
    dpf = st.number_input("DPF", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 32)

    if st.button("🔍 Predict Diabetes"):
        X = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
        X = scaler.transform(X)
        pred = model.predict(X)[0]
        st.error("⚠️ Diabetes Detected" if pred == 1 else "✅ No Diabetes")

# ================= BRAIN TUMOR =================
elif disease == "Brain Tumor":
    st.subheader("🧠 Brain Tumor Prediction (MRI Image)")
    model = load_brain_model()

    file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_column_width=True)

        img = image.resize((128,128))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        if st.button("🔍 Predict Brain Tumor"):
            pred = model.predict(img)[0][0]
            st.error("⚠️ Brain Tumor Detected" if pred > 0.5 else "✅ No Brain Tumor")

st.markdown("---")
st.markdown("Made with ❤️ for ML Deployment")
