import streamlit as st
import pickle
import pandas as pd

# =========================
# Custom CSS
# =========================
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f4f8;
        padding: 20px;
        border-right: 2px solid #d0d7de;
    }
    h1 {
        color: #2c3e50;
        font-size: 2.2em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }
    .stTextInput > div > input,
    .stNumberInput > div > input,
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 8px;
        font-size: 16px;
    }
    button[kind="primary"] {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 20px;
    }
    button[kind="primary"]:hover {
        background-color: #27ae60;
        color: white;
    }
    .stAlert-success {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Load models
# =========================
MODELS = {
    "Kidney": "models/kidney_model.pkl",
    "Liver": "models/liver_model.pkl",
    "Heart": "models/heart_model.pkl",
    "Diabetes": "models/diabetes_model.pkl"
}

# Manual feature schemas (must match training dataset order!)
FEATURES = {
    "Kidney": ["age", "bp", "sg", "al", "su"],  # extend if trained with more features
    "Liver": ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
              "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins",
              "Albumin", "Albumin_and_Globulin_Ratio"],
    "Heart": ["Age", "Sex", "Chest pain type", "BP", "Cholesterol",
              "FBS over 120", "EKG results", "Max HR", "Exercise angina",
              "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"],
    "Diabetes": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
}

st.title("🩺 Multi-Disease Prediction Portal")
st.write("Select a disease type, enter patient details, and get prediction results.")

# =========================
# Disease selector
# =========================
disease = st.selectbox("Choose a disease to predict:", list(MODELS.keys()))

# Load chosen model
with open(MODELS[disease], "rb") as f:
    model, scaler = pickle.load(f)

# =========================
# Input forms per disease
# =========================
st.subheader(f"{disease} Disease Input Form")

input_data = {}

if disease == "Kidney":
    input_data["age"] = st.number_input("Age", 1, 120, 48)
    input_data["bp"] = st.number_input("Blood Pressure", 50, 200, 80)
    input_data["sg"] = st.number_input("Specific Gravity", 1.0, 1.05, 1.02)
    input_data["al"] = st.number_input("Albumin", 0, 5, 1)
    input_data["su"] = st.number_input("Sugar", 0, 5, 0)

elif disease == "Liver":
    input_data["Age"] = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    # Encode gender as numeric (must match training)
    input_data["Gender"] = 1 if gender == "Male" else 0
    input_data["Total_Bilirubin"] = st.number_input("Total Bilirubin", 0.0, 10.0, 1.3)
    input_data["Direct_Bilirubin"] = st.number_input("Direct Bilirubin", 0.0, 5.0, 0.4)
    input_data["Alkaline_Phosphotase"] = st.number_input("Alkaline Phosphotase", 50, 2000, 210)
    input_data["Alamine_Aminotransferase"] = st.number_input("Alamine Aminotransferase", 1, 2000, 35)
    input_data["Aspartate_Aminotransferase"] = st.number_input("Aspartate Aminotransferase", 1, 2000, 40)
    input_data["Total_Proteins"] = st.number_input("Total Proteins", 1.0, 10.0, 6.8)
    input_data["Albumin"] = st.number_input("Albumin", 1.0, 6.0, 3.1)
    input_data["Albumin_and_Globulin_Ratio"] = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0, 0.9)

elif disease == "Heart":
    input_data["Age"] = st.number_input("Age", 1, 120, 52)
    input_data["Sex"] = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    input_data["Chest pain type"] = st.number_input("Chest Pain Type", 0, 4, 0)
    input_data["BP"] = st.number_input("Blood Pressure", 50, 200, 120)
    input_data["Cholesterol"] = st.number_input("Cholesterol", 100, 600, 240)
    input_data["FBS over 120"] = st.selectbox("FBS > 120", [0, 1])
    input_data["EKG results"] = st.number_input("EKG Results", 0, 2, 1)
    input_data["Max HR"] = st.number_input("Max HR", 60, 250, 150)
    input_data["Exercise angina"] = st.selectbox("Exercise Angina", [0, 1])
    input_data["ST depression"] = st.number_input("ST Depression", 0.0, 10.0, 1.2)
    input_data["Slope of ST"] = st.number_input("Slope of ST", 0, 3, 2)
    input_data["Number of vessels fluro"] = st.number_input("Number of vessels fluro", 0, 4, 0)
    input_data["Thallium"] = st.number_input("Thallium", 0, 7, 2)

elif disease == "Diabetes":
    input_data["Pregnancies"] = st.number_input("Pregnancies", 0, 20, 2)
    input_data["Glucose"] = st.number_input("Glucose", 0, 300, 120)
    input_data["BloodPressure"] = st.number_input("Blood Pressure", 0, 200, 70)
    input_data["SkinThickness"] = st.number_input("Skin Thickness", 0, 100, 20)
    input_data["Insulin"] = st.number_input("Insulin", 0, 900, 85)
    input_data["BMI"] = st.number_input("BMI", 0.0, 70.0, 28.5)
    input_data["DiabetesPedigreeFunction"] = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    input_data["Age"] = st.number_input("Age", 1, 120, 32)

# =========================
# Prediction
# =========================
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=FEATURES[disease], fill_value=0)

    # FIX: use .values to avoid feature name mismatch error
    input_scaled = scaler.transform(input_df.values)

    pred = model.predict(input_scaled)[0]
    st.success(f"Prediction: {'Disease Detected' if pred == 1 else 'No Disease'}")
