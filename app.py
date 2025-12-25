import streamlit as st
import numpy as np
import pickle
from fpdf import FPDF
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import io

# ===================== SESSION INIT =====================
if 'page' not in st.session_state:
    st.session_state['page'] = 'Signup'
if 'users' not in st.session_state:
    st.session_state['users'] = {}
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None
if 'report' not in st.session_state:
    st.session_state['report'] = ""

# ===================== PDF CREATOR =====================
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    safe_text = text.encode("latin1", "ignore").decode("latin1")
    pdf.multi_cell(0, 10, safe_text)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

# ===================== MODEL LOADERS =====================
@st.cache_resource
def load_pickle_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        model, scaler = data
    else:
        model = data['model']
        scaler = data['scaler']
    return model, scaler

@st.cache_resource
def load_brain_model():
    FILE_ID = "1r7Kmf14ZGKQK3GSTk3nxPxfAyGpg2m_b"  # Replace with your file ID
    URL = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
    response = requests.get(URL)
    with open("brain_tumor_dataset.h5", "wb") as f:
        f.write(response.content)
    model = load_model("brain_tumor_dataset.h5")
    return model

# ===================== SIGNUP =====================
def signup():
    st.title("📝 Signup")
    username = st.text_input("Enter username")
    password = st.text_input("Enter password", type="password")
    if st.button("Signup"):
        if username in st.session_state['users']:
            st.error("Username already exists!")
        elif username == "" or password == "":
            st.error("Please enter valid credentials")
        else:
            st.session_state['users'][username] = password
            st.success("Signup successful! Please login.")
            st.session_state['page'] = 'Login'

# ===================== LOGIN =====================
def login():
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state['users'] and st.session_state['users'][username] == password:
            st.session_state['logged_in'] = True
            st.session_state['current_user'] = username
            st.session_state['page'] = 'Home'
        else:
            st.error("Invalid username or password")

# ===================== HOME DASHBOARD =====================
def home_dashboard():
    st.title("🩺 Multi-Disease Diagnostic Portal")
    st.write(f"Welcome **{st.session_state['current_user']}**! Select a disease below:")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("❤️ Heart"): st.session_state['page'] = 'Heart'
    with col2:
        if st.button("🩸 Diabetes"): st.session_state['page'] = 'Diabetes'
    with col3:
        if st.button("🧠 Brain Tumor"): st.session_state['page'] = 'Brain'
    col4, col5 = st.columns(2)
    with col4:
        if st.button("🟣 Kidney"): st.session_state['page'] = 'Kidney'
    with col5:
        if st.button("🟠 Liver"): st.session_state['page'] = 'Liver'

    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.session_state['page'] = 'Login'

# ===================== PREDICTION PAGES =====================
def disease_page(title, model_loader, input_func=None, is_brain=False):
    st.header(f"{title} Prediction")
    inputs = None
    image = None
    if is_brain:
        uploaded_file = st.file_uploader("Upload MRI Image...", type=["jpg","jpeg","png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI", use_column_width=True)
    else:
        if input_func:
            inputs = input_func()

    if st.button(f"🔍 Predict {title}"):
        try:
            if is_brain:
                model = model_loader()
                # Preprocess
                input_shape = model.input_shape[1:]
                if len(input_shape)==1:
                    side = int(np.sqrt(input_shape[0]/3))
                    img = image.resize((side, side))
                    X = np.array(img)/255.0
                    X = X.flatten().reshape(1,-1)
                else:
                    img = image.resize((input_shape[0], input_shape[1]))
                    X = np.array(img)/255.0
                    X = np.expand_dims(X,0)
                pred = model.predict(X)[0][0]
                result_text = f"{title} Result: {'⚠️ Detected' if pred>0.5 else '✅ Not Detected'}"
            else:
                model, scaler = model_loader()
                X = np.array([inputs])
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]
                result_text = f"{title} Result: {'⚠️ Detected' if pred==1 else '✅ Not Detected'}"

            if '⚠️' in result_text:
                st.error(result_text)
            else:
                st.success(result_text)

            # PDF report
            st.session_state['report'] = f"User: {st.session_state['current_user']}\n{result_text}"
            pdf_bytes = create_pdf(st.session_state['report'])
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{title.replace(' ','_')}_Report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

    st.button("⬅️ Back", on_click=lambda: st.session_state.update({'page':'Home'}))

# ===================== INPUT FUNCTIONS =====================
def heart_inputs():
    age = st.number_input("Age", 0, 120, 52)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
    cp = st.number_input("Chest Pain Type (0-3)", 0,3,0)
    trestbps = st.number_input("BP", 80,200,120)
    chol = st.number_input("Cholesterol", 100,600,240)
    fbs = st.selectbox("FBS > 120", [0,1])
    restecg = st.number_input("Rest ECG (0-2)", 0,2,1)
    thalach = st.number_input("Max HR", 60,250,150)
    exang = st.selectbox("Exercise angina", [0,1])
    oldpeak = st.number_input("ST Depression",0.0,10.0,1.2)
    slope = st.number_input("Slope ST",0,2,1)
    ca = st.number_input("Number of vessels",0,3,0)
    thal = st.number_input("Thalassemia (1,2,3)",1,3,2)
    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

def diabetes_inputs():
    preg = st.number_input("Pregnancies",0,20,2)
    glucose = st.number_input("Glucose",0,300,120)
    bp = st.number_input("BP",0,200,70)
    skin = st.number_input("Skin Thickness",0,100,20)
    insulin = st.number_input("Insulin",0,900,85)
    bmi = st.number_input("BMI",0.0,70.0,28.5)
    dpf = st.number_input("DPF",0.0,3.0,0.5)
    age = st.number_input("Age",1,120,32)
    return [preg, glucose, bp, skin, insulin, bmi, dpf, age]

def kidney_inputs():
    age = st.number_input("Age", 0, 120, 45)
    bp = st.number_input("Blood Pressure", 0, 200, 80)
    sg = st.number_input("Specific Gravity", 1.0, 1.05, 1.020)
    al = st.number_input("Albumin", 0,5,0)
    su = st.number_input("Sugar",0,5,0)
    bgr = st.number_input("Blood Glucose Random",0,500,110)
    bu = st.number_input("Blood Urea",0,200,25)
    sc = st.number_input("Serum Creatinine",0.0,20.0,1.0)
    hemo = st.number_input("Hemoglobin",0.0,20.0,15.2)
    pcv = st.number_input("Packed Cell Volume",0,60,44)
    return [age, bp, sg, al, su, bgr, bu, sc, hemo, pcv]

def liver_inputs():
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_val = 1 if gender=="Male" else 0
    total_bilirubin = st.number_input("Total Bilirubin",0.0,10.0,1.3)
    direct_bilirubin = st.number_input("Direct Bilirubin",0.0,5.0,0.4)
    alk_phos = st.number_input("Alkaline Phosphotase",50,2000,210)
    alt = st.number_input("ALT",1,2000,35)
    ast = st.number_input("AST",1,2000,40)
    total_proteins = st.number_input("Total Proteins",1.0,10.0,6.8)
    albumin = st.number_input("Albumin",1.0,6.0,3.1)
    ag_ratio = st.number_input("Albumin/Globulin Ratio",0.0,3.0,0.9)
    return [age, gender_val, total_bilirubin, direct_bilirubin, alk_phos, alt, ast, total_proteins, albumin, ag_ratio]

# ===================== MAIN =====================
if st.session_state['page'] == 'Signup':
    signup()
elif st.session_state['page'] == 'Login':
    login()
elif st.session_state['page'] == 'Home':
    home_dashboard()
elif st.session_state['page'] == 'Heart':
    disease_page("Heart Disease", lambda: load_pickle_model("models/heart_model.pkl"), heart_inputs)
elif st.session_state['page'] == 'Diabetes':
    disease_page("Diabetes", lambda: load_pickle_model("models/diabetes_model.pkl"), diabetes_inputs)
elif st.session_state['page'] == 'Kidney':
    disease_page("Kidney Disease", lambda: load_pickle_model("models/kidney_10f_model.pkl"), kidney_inputs)
elif st.session_state['page'] == 'Liver':
    disease_page("Liver Disease", lambda: load_pickle_model("models/liver_model.pkl"), liver_inputs)
elif st.session_state['page'] == 'Brain':
    disease_page("Brain Tumor", load_brain_model, is_brain=True)
