import streamlit as st
import numpy as np
import pickle
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import requests
import tempfile
import speech_recognition as sr
from tensorflow.keras.models import load_model
import pandas as pd

# ===================== SESSION INIT =====================
if 'page' not in st.session_state:
    st.session_state.page = "Login"
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'appointments' not in st.session_state:
    st.session_state.appointments = {}

# ===================== PDF REPORT =====================
def create_pdf(username, disease, result, input_data, image=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, 15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Multi Disease Diagnostic Report", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Patient: {username}", ln=True)
    pdf.cell(0, 8, f"Disease: {disease}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Input Summary", ln=True)

    pdf.set_font("Arial", size=11)
    for k, v in input_data.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Prediction Result", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, result)

    if image:
        img_path = "temp_img.jpg"
        image.save(img_path)
        pdf.ln(5)
        pdf.image(img_path, x=30, w=150)

    return pdf.output(dest="S").encode("latin1")

# ===================== MODEL LOADERS =====================
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        model, scaler = pickle.load(f)
    return model, scaler

@st.cache_resource
def load_brain_model():
    return load_model("models/brain_tumor_model.h5")

# ===================== COMMON UI =====================
def show_table(data):
    st.subheader("📋 Patient Input Summary")
    st.table(pd.DataFrame(data.items(), columns=["Parameter", "Value"]))

# ===================== AUTH =====================
def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in st.session_state.users and st.session_state.users[u] == p:
            st.session_state.logged_in = True
            st.session_state.current_user = u
            st.session_state.page = "Home"
        else:
            st.error("Invalid credentials")

def signup():
    st.title("📝 Signup")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Signup"):
        st.session_state.users[u] = p
        st.success("Signup successful")
        st.session_state.page = "Login"

# ===================== HOME =====================
def home():
    st.title("🩺 Multi Disease Diagnostic Platform")
    options = ["Heart", "Diabetes", "Kidney", "Liver", "Brain"]
    choice = st.selectbox("Select Disease", options)
    if st.button("Proceed"):
        st.session_state.page = choice
    if st.button("Logout"):
        st.session_state.page = "Login"
        st.session_state.logged_in = False

# ===================== HEART =====================
def heart():
    st.header("❤️ Heart Disease")
    data = {
        "Age": st.number_input("Age", 1, 120),
        "Sex": st.selectbox("Sex", [0,1]),
        "Chest Pain": st.number_input("Chest Pain",0,3),
        "BP": st.number_input("Blood Pressure"),
        "Cholesterol": st.number_input("Cholesterol"),
        "FBS": st.selectbox("FBS", [0,1]),
        "ECG": st.number_input("ECG"),
        "Max HR": st.number_input("Max HR"),
        "Angina": st.selectbox("Angina", [0,1]),
        "ST Depression": st.number_input("ST Depression"),
        "Slope": st.number_input("Slope"),
        "Vessels": st.number_input("Vessels"),
        "Thal": st.number_input("Thal")
    }
    show_table(data)

    if st.button("Predict"):
        model, scaler = load_pickle("models/heart_model.pkl")
        X = scaler.transform([list(data.values())])
        pred = model.predict(X)[0]
        result = "⚠️ Heart Disease Detected" if pred else "✅ No Heart Disease"
        st.success(result)

        pdf = create_pdf(st.session_state.current_user,"Heart Disease",result,data)
        st.download_button("Download Report", pdf, "heart_report.pdf")

# ===================== DIABETES =====================
def diabetes():
    st.header("🩸 Diabetes")
    data = {
        "Pregnancies": st.number_input("Pregnancies"),
        "Glucose": st.number_input("Glucose"),
        "BP": st.number_input("BP"),
        "Skin": st.number_input("Skin Thickness"),
        "Insulin": st.number_input("Insulin"),
        "BMI": st.number_input("BMI"),
        "DPF": st.number_input("DPF"),
        "Age": st.number_input("Age")
    }
    show_table(data)

    if st.button("Predict"):
        model, scaler = load_pickle("models/diabetes_model.pkl")
        X = scaler.transform([list(data.values())])
        pred = model.predict(X)[0]
        result = "⚠️ Diabetes Detected" if pred else "✅ No Diabetes"
        st.success(result)

        pdf = create_pdf(st.session_state.current_user,"Diabetes",result,data)
        st.download_button("Download Report", pdf, "diabetes_report.pdf")

# ===================== BRAIN =====================
def brain():
    st.header("🧠 Brain Tumor")
    model = load_brain_model()
    file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).resize((224,224))
        st.image(img)

        data = {
            "File": file.name,
            "Size": img.size,
            "Mode": img.mode
        }
        show_table(data)

        if st.button("Predict"):
            arr = np.array(img)/255.0
            arr = arr.reshape(1,224,224,3)
            pred = model.predict(arr)[0][0]
            result = "⚠️ Brain Tumor Detected" if pred > 0.5 else "✅ No Brain Tumor"
            st.success(result)

            pdf = create_pdf(
                st.session_state.current_user,
                "Brain Tumor",
                result,
                data,
                image=img
            )
            st.download_button("Download Report", pdf, "brain_report.pdf")

# ===================== ROUTER =====================
if st.session_state.page == "Login":
    login()
elif st.session_state.page == "Signup":
    signup()
elif st.session_state.page == "Home":
    home()
elif st.session_state.page == "Heart":
    heart()
elif st.session_state.page == "Diabetes":
    diabetes()
elif st.session_state.page == "Brain":
    brain()
