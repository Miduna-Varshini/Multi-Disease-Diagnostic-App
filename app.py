import streamlit as st
import pickle
import pandas as pd

# =========================
# Custom CSS
# =========================
st.markdown(
    """
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
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Model paths
# =========================
MODELS = {
    "Kidney": "models/kidney_model.pkl",
    "Liver": "models/liver_model.pkl",
    "Heart": "models/heart_model.pkl",
    "Diabetes": "models/diabetes_model.pkl",
}

# =========================
# Feature schemas (order must match training)
# =========================
FEATURES = {
    # CKD full feature set (common Kaggle/UCI schema)
    "Kidney": [
        "age", "bp", "sg", "al", "su",
        "rbc", "pc", "pcc", "ba",
        "bgr", "bu", "sc", "sod", "pot",
        "hemo", "pcv", "wbcc", "rbcc",
        "htn", "dm", "cad", "appet", "pe", "ane",
    ],
    "Liver": [
        "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Proteins",
        "Albumin", "Albumin_and_Globulin_Ratio",
    ],
    "Heart": [
        "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
        "FBS over 120", "EKG results", "Max HR", "Exercise angina",
        "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium",
    ],
    "Diabetes": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ],
}

# =========================
# App header
# =========================
st.title("🩺 Multi-Disease Prediction Portal")
st.write("Select a disease type, enter patient details, and get prediction results.")

# =========================
# Disease selector and model loader
# =========================
disease = st.selectbox("Choose a disease to predict:", list(MODELS.keys()))
with open(MODELS[disease], "rb") as f:
    model, scaler = pickle.load(f)

# =========================
# Input forms
# =========================
st.subheader(f"{disease} Disease Input Form")
input_data = {}

if disease == "Kidney":
    # Numeric basics
    input_data["age"] = st.number_input("Age", 1, 120, 48)
    input_data["bp"] = st.number_input("Blood Pressure", 50, 200, 80)
    input_data["sg"] = st.number_input("Specific Gravity", 1.0, 1.05, 1.02, format="%.3f")
    input_data["al"] = st.number_input("Albumin", 0, 5, 1)
    input_data["su"] = st.number_input("Sugar", 0, 5, 0)

    # Categorical (string)
    input_data["rbc"] = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    input_data["pc"] = st.selectbox("Pus Cell", ["normal", "abnormal"])
    input_data["pcc"] = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
    input_data["ba"] = st.selectbox("Bacteria", ["present", "notpresent"])

    # Numeric lab values
    input_data["bgr"] = st.number_input("Blood Glucose Random", 0.0, 500.0, 121.0)
    input_data["bu"] = st.number_input("Blood Urea", 0.0, 200.0, 36.0)
    input_data["sc"] = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2)
    input_data["sod"] = st.number_input("Sodium", 0.0, 200.0, 138.0)
    input_data["pot"] = st.number_input("Potassium", 0.0, 10.0, 4.5)
    input_data["hemo"] = st.number_input("Hemoglobin", 0.0, 20.0, 15.0)
    input_data["pcv"] = st.number_input("Packed Cell Volume", 0.0, 60.0, 44.0)
    input_data["wbcc"] = st.number_input("White Blood Cell Count", 0.0, 20000.0, 7800.0)
    input_data["rbcc"] = st.number_input("Red Blood Cell Count", 0.0, 10.0, 5.2)

    # Categorical (string yes/no or appetite)
    input_data["htn"] = st.selectbox("Hypertension", ["yes", "no"])
    input_data["dm"] = st.selectbox("Diabetes Mellitus", ["yes", "no"])
    input_data["cad"] = st.selectbox("Coronary Artery Disease", ["yes", "no"])
    input_data["appet"] = st.selectbox("Appetite", ["good", "poor"])
    input_data["pe"] = st.selectbox("Pedal Edema", ["yes", "no"])
    input_data["ane"] = st.selectbox("Anemia", ["yes", "no"])

elif disease == "Liver":
    input_data["Age"] = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
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
    input_data["Sex"] = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
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
    # Encode categorical Kidney fields to match training (string -> numeric)
    if disease == "Kidney":
        categorical_map = {
            "normal": 1, "abnormal": 0,
            "present": 1, "notpresent": 0,
            "yes": 1, "no": 0,
            "good": 1, "poor": 0,
        }
        for key, val in list(input_data.items()):
            if isinstance(val, str) and val in categorical_map:
                input_data[key] = categorical_map[val]

    # Align to fixed schema and fill missing with zeros
    input_df = pd.DataFrame([input_data]).reindex(columns=FEATURES[disease], fill_value=0)

    # Use raw values so scaler ignores column names (prevents feature-name ValueError)
    input_scaled = scaler.transform(input_df.values)

    # Predict and display result
    pred = int(model.predict(input_scaled)[0])
    st.success(f"Prediction: {'Disease Detected' if pred == 1 else 'No Disease'}")
