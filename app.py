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

    # flex-box like layout using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("❤️ Heart"):
            st.session_state['page'] = 'Heart'
    with col2:
        if st.button("🩸 Diabetes"):
            st.session_state['page'] = 'Diabetes'
    with col3:
        if st.button("🧠 Brain"):
            st.session_state['page'] = 'Brain'
    col4, col5 = st.columns(2)
    with col4:
        if st.button("🟣 Kidney"):
            st.session_state['page'] = 'Kidney'
    with col5:
        if st.button("🟠 Liver"):
            st.session_state['page'] = 'Liver'

    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.session_state['page'] = 'Login'

# ===================== PREDICTION PAGES =====================
def heart_page():
    st.header("❤️ Heart Disease Prediction")
    age = st.number_input("Age", 0, 120, 52)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
    trestbps = st.number_input("BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 240)
    fbs = st.selectbox("FBS > 120", [0,1])
    restecg = st.number_input("Rest ECG (0-2)", 0,2,1)
    thalach = st.number_input("Max HR", 60,250,150)
    exang = st.selectbox("Exercise angina", [0,1])
    oldpeak = st.number_input("ST Depression",0.0,10.0,1.2)
    slope = st.number_input("Slope ST",0,2,1)
    ca = st.number_input("Number of vessels",0,3,0)
    thal = st.number_input("Thalassemia (1,2,3)",1,3,2)

    if st.button("Predict Heart Disease"):
        model, scaler, FEATURES = load_model("models/heart_model.pkl", [])
        X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        if pred==1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease")

    st.button("Back", on_click=lambda: st.session_state.update({'page':'Home'}))

# ===================== OTHER PREDICTION PAGES =====================
def diabetes_page():
    st.header("🩸 Diabetes Prediction")
    preg = st.number_input("Pregnancies",0,20,2)
    glucose = st.number_input("Glucose",0,300,120)
    bp = st.number_input("BP",0,200,70)
    skin = st.number_input("Skin Thickness",0,100,20)
    insulin = st.number_input("Insulin",0,900,85)
    bmi = st.number_input("BMI",0.0,70.0,28.5)
    dpf = st.number_input("DPF",0.0,3.0,0.5)
    age = st.number_input("Age",1,120,32)

    if st.button("Predict Diabetes"):
        model, scaler, FEATURES = load_model("models/diabetes_model.pkl", [])
        X = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        if pred==1:
            st.error("⚠️ Diabetes Detected")
        else:
            st.success("✅ No Diabetes")

    st.button("Back", on_click=lambda: st.session_state.update({'page':'Home'}))

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
# You can similarly add Brain, Kidney, Liver pages
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

    # flex-box like layout using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("❤️ Heart"):
            st.session_state['page'] = 'Heart'
    with col2:
        if st.button("🩸 Diabetes"):
            st.session_state['page'] = 'Diabetes'
    with col3:
        if st.button("🧠 Brain"):
            st.session_state['page'] = 'Brain'
    col4, col5 = st.columns(2)
    with col4:
        if st.button("🟣 Kidney"):
            st.session_state['page'] = 'Kidney'
    with col5:
        if st.button("🟠 Liver"):
            st.session_state['page'] = 'Liver'

    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.session_state['page'] = 'Login'

# ===================== PREDICTION PAGES =====================
def heart_page():
    st.header("❤️ Heart Disease Prediction")
    age = st.number_input("Age", 0, 120, 52)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
    trestbps = st.number_input("BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 240)
    fbs = st.selectbox("FBS > 120", [0,1])
    restecg = st.number_input("Rest ECG (0-2)", 0,2,1)
    thalach = st.number_input("Max HR", 60,250,150)
    exang = st.selectbox("Exercise angina", [0,1])
    oldpeak = st.number_input("ST Depression",0.0,10.0,1.2)
    slope = st.number_input("Slope ST",0,2,1)
    ca = st.number_input("Number of vessels",0,3,0)
    thal = st.number_input("Thalassemia (1,2,3)",1,3,2)

    if st.button("Predict Heart Disease"):
        model, scaler, FEATURES = load_model("models/heart_model.pkl", [])
        X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        if pred==1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease")

    st.button("Back", on_click=lambda: st.session_state.update({'page':'Home'}))

# ===================== OTHER PREDICTION PAGES =====================
def diabetes_page():
    st.header("🩸 Diabetes Prediction")
    preg = st.number_input("Pregnancies",0,20,2)
    glucose = st.number_input("Glucose",0,300,120)
    bp = st.number_input("BP",0,200,70)
    skin = st.number_input("Skin Thickness",0,100,20)
    insulin = st.number_input("Insulin",0,900,85)
    bmi = st.number_input("BMI",0.0,70.0,28.5)
    dpf = st.number_input("DPF",0.0,3.0,0.5)
    age = st.number_input("Age",1,120,32)

    if st.button("Predict Diabetes"):
        model, scaler, FEATURES = load_model("models/diabetes_model.pkl", [])
        X = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        if pred==1:
            st.error("⚠️ Diabetes Detected")
        else:
            st.success("✅ No Diabetes")

    st.button("Back", on_click=lambda: st.session_state.update({'page':'Home'}))

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
# You can similarly add Brain, Kidney, Liver pages
