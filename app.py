import streamlit as st
import pickle
import numpy as np

# ===================== SESSION INIT =====================
if 'page' not in st.session_state:
    st.session_state['page'] = 'Signup'
if 'users' not in st.session_state:
    st.session_state['users'] = {}  # simple user storage
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None

# ===================== MODEL LOADER =====================
@st.cache_resource
def load_model(path, default_features=[]):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        model, scaler = data
        features = default_features
    else:
        model = data['model']
        scaler = data['scaler']
        features = data['features']
    return model, scaler, features

# ===================== SIGNUP =====================
def signup():
    st.title("📝 Signup")
    username = st.text_input("Enter username", key="signup_user")
    password = st.text_input("Enter password", type="password", key="signup_pass")
    if st.button("Signup", key="signup_btn"):
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
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login", key="login_btn"):
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

    # Flexbox-like layout using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("❤️ Heart", key="btn_heart"):
            st.session_state['page'] = 'Heart'
    with col2:
        if st.button("🩸 Diabetes", key="btn_diabetes"):
            st.session_state['page'] = 'Diabetes'
    with col3:
        if st.button("🧠 Brain", key="btn_brain"):
            st.session_state['page'] = 'Brain'

    col4, col5 = st.columns(2)
    with col4:
        if st.button("🟣 Kidney", key="btn_kidney"):
            st.session_state['page'] = 'Kidney'
    with col5:
        if st.button("🟠 Liver", key="btn_liver"):
            st.session_state['page'] = 'Liver'

    if st.button("Logout", key="btn_logout"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.session_state['page'] = 'Login'

# ===================== HEART PAGE =====================
def heart_page():
    st.header("❤️ Heart Disease Prediction")
    age = st.number_input("Age", 0, 120, 52, key="heart_age")
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1], key="heart_sex")
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0, key="heart_cp")
    trestbps = st.number_input("BP", 80, 200, 120, key="heart_bp")
    chol = st.number_input("Cholesterol", 100, 600, 240, key="heart_chol")
    fbs = st.selectbox("FBS > 120", [0,1], key="heart_fbs")
    restecg = st.number_input("Rest ECG (0-2)", 0,2,1, key="heart_restecg")
    thalach = st.number_input("Max HR", 60,250,150, key="heart_thalach")
    exang = st.selectbox("Exercise angina", [0,1], key="heart_exang")
    oldpeak = st.number_input("ST Depression",0.0,10.0,1.2, key="heart_oldpeak")
    slope = st.number_input("Slope ST",0,2,1, key="heart_slope")
    ca = st.number_input("Number of vessels",0,3,0, key="heart_ca")
    thal = st.number_input("Thalassemia (1,2,3)",1,3,2, key="heart_thal")

    if st.button("Predict Heart Disease", key="heart_predict"):
        model, scaler, FEATURES = load_model("models/heart_model.pkl", [])
        X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        if pred==1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease")

    st.button("Back", key="heart_back", on_click=lambda: st.session_state.update({'page':'Home'}))

# ===================== DIABETES PAGE =====================
def diabetes_page():
    st.header("🩸 Diabetes Prediction")
    preg = st.number_input("Pregnancies",0,20,2, key="dia_preg")
    glucose = st.number_input("Glucose",0,300,120, key="dia_glucose")
    bp = st.number_input("BP",0,200,70, key="dia_bp")
    skin = st.number_input("Skin Thickness",0,100,20, key="dia_skin")
    insulin = st.number_input("Insulin",0,900,85, key="dia_insulin")
    bmi = st.number_input("BMI",0.0,70.0,28.5, key="dia_bmi")
    dpf = st.number_input("DPF",0.0,3.0,0.5, key="dia_dpf")
    age = st.number_input("Age",1,120,32, key="dia_age")

    if st.button("Predict Diabetes", key="dia_predict"):
        model, scaler, FEATURES = load_model("models/diabetes_model.pkl", [])
        X = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        if pred==1:
            st.error("⚠️ Diabetes Detected")
        else:
            st.success("✅ No Diabetes")

    st.button("Back", key="dia_back", on_click=lambda: st.session_state.update({'page':'Home'}))

# ===================== PLACEHOLDER PAGES =====================
def brain_page():
    st.header("🧠 Brain Tumor Detection")
    st.info("You can integrate your brain tumor .h5 model here.")
    st.button("Back", key="brain_back", on_click=lambda: st.session_state.update({'page':'Home'}))

def kidney_page():
    st.header("🟣 Kidney Disease Prediction")
    st.info("You can integrate your kidney .pkl model here.")
    st.button("Back", key="kidney_back", on_click=lambda: st.session_state.update({'page':'Home'}))

def liver_page():
    st.header("🟠 Liver Disease Prediction")
    st.info("You can integrate your liver .pkl model here.")
    st.button("Back", key="liver_back", on_click=lambda: st.session_state.update({'page':'Home'}))

# ===================== MAIN PAGE CONTROL =====================
if st.session_state['page'] == 'Signup':
    signup()
elif st.session_state['page'] == 'Login':
    login()
elif st.session_state['page'] == 'Home':
    home_dashboard()
elif st.session_state['page'] == 'Heart':
    heart_page()
elif st.session_state['page'] == 'Diabetes':
    diabetes_page()
elif st.session_state['page'] == 'Brain':
    brain_page()
elif st.session_state['page'] == 'Kidney':
    kidney_page()
elif st.session_state['page'] == 'Liver':
    liver_page()
