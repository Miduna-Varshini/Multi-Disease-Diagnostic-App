import streamlit as st
import numpy as np
import pickle
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import speech_recognition as sr
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

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
if 'appointments' not in st.session_state:
    st.session_state['appointments'] = {}

# ===================== STYLES =====================
st.markdown("""
<style>
.stApp {background-color:white;}
.dashboard-container {display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:20px; padding:20px;}
.card {
    width:100%; height:120px; border-radius:15px; color:black;
    background-color:#4ade80; border:2px solid red; padding:20px;
    font-family:Arial,sans-serif; display:flex; flex-direction:column;
    justify-content:center; text-align:left; cursor:pointer;
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    box-shadow:0px 8px 20px rgba(0,0,0,0.4);
}
.card:hover {transform:scale(1.03); box-shadow:0px 15px 30px rgba(0,0,0,0.5);}
.card-title {font-size:22px; font-weight:bold; margin-bottom:5px;}
.card-subtitle {font-size:14px; opacity:0.9;}
.logout-btn {
    background-color:red; color:white; border-radius:10px; padding:15px;
    font-weight:bold; width:100%; border:none; cursor:pointer; text-align:center;
    box-shadow:0px 8px 20px rgba(0,0,0,0.4);
}
.logout-btn:hover {background-color:#b91c1c;}
</style>
""", unsafe_allow_html=True)

# ===================== PDF CREATOR =====================
def create_pdf(username, disease, result_text, image=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Multi Disease Diagnostic Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    login_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")
    content = f"Username        : {username}\nLogin Time     : {login_time}\nDisease        : {disease}\n\nPrediction Result:\n{result_text}"
    safe_text = content.encode("latin1","ignore").decode("latin1")
    pdf.multi_cell(0,8,safe_text)
    pdf.ln(5)
    if image:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Uploaded MRI Image:", ln=True)
        img_path = "temp_brain_image.jpg"
        image.save(img_path)
        pdf.image(img_path, x=30, w=150)
        pdf.ln(10)
    return pdf.output(dest="S").encode("latin1")

# ===================== MODEL LOADERS =====================
@st.cache_resource
def load_pickle_model(path):
    with open(path,"rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        model, scaler = data
    else:
        model = data['model']
        scaler = data['scaler']
    return model, scaler

@st.cache_resource
def load_brain_model(path="models/brain_tumor_model.h5"):
    model = load_model(path)
    return model

# ===================== SIGNUP & LOGIN =====================
def signup():
    st.title("📝 Signup")
    username = st.text_input("Enter username")
    password = st.text_input("Enter password", type="password")
    if st.button("Signup"):
        if username in st.session_state['users']:
            st.error("Username already exists!")
        elif username=="" or password=="":
            st.error("Enter valid credentials")
        else:
            st.session_state['users'][username] = password
            st.success("Signup successful! Please login.")
            st.session_state['page'] = 'Login'

def login():
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state['users'] and st.session_state['users'][username]==password:
            st.session_state['logged_in'] = True
            st.session_state['current_user'] = username
            st.session_state['page'] = 'Home'
        else:
            st.error("Invalid username or password")

# ===================== HOME DASHBOARD =====================
def home_dashboard():
    st.markdown(f"<h1 style='text-align:center; color:black'>🩺 Multi-Disease Diagnostic Portal</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center; color:black; margin-bottom:30px;'>Welcome <b>{st.session_state['current_user']}</b>! Select a disease:</h3>", unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    cards = [
        ("❤️ Heart","heart_card","Predict Heart Disease","Heart"),
        ("🩸 Diabetes","diabetes_card","Predict Diabetes","Diabetes"),
        ("🧠 Brain Tumor","brain_card","Predict Brain Tumor","Brain"),
        ("🟣 Kidney","kidney_card","Predict Kidney Disease","Kidney"),
        ("🟠 Liver","liver_card","Predict Liver Disease","Liver"),
        ("🎙️ Speech to Text","speech_card","Voice Input","Speech")
    ]
    for title,key,subtitle,page in cards:
        if st.button(title,key=key):
            st.session_state['page'] = page
        st.markdown(f'<div class="card"><div class="card-title">{title}</div><div class="card-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)
    
    if st.button("Logout", key="logout_card"):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.session_state['page'] = 'Login'
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== DISEASE INPUTS =====================
# ===================== GENERIC DISEASE PAGE =====================
def disease_page(disease_name, model_loader, input_func):
    st.header(f"🧪 {disease_name} Prediction")

    inputs = input_func()

    if st.button("🔍 Predict"):
        try:
            model, scaler = model_loader()

            X = np.array(inputs).reshape(1, -1)
            X_scaled = scaler.transform(X)

            prediction = model.predict(X_scaled)[0]

            if prediction == 1:
                result_text = f"⚠️ {disease_name} Detected"
                st.error(result_text)
            else:
                result_text = f"✅ No {disease_name} Detected"
                st.success(result_text)

            # PDF
            pdf_bytes = create_pdf(
                username=st.session_state['current_user'],
                disease=disease_name,
                result_text=result_text
            )
            st.download_button(
                "📄 Download PDF Report",
                pdf_bytes,
                f"{disease_name}_Report.pdf",
                "application/pdf"
            )

            # Appointment + Hospitals
            appointment_booking(disease_name)
            show_hospitals(disease_name)

        except Exception as e:
            st.error("Prediction failed ❌")
            st.code(str(e))

    st.button("⬅️ Back", on_click=lambda: st.session_state.update({'page': 'Home'}))

def heart_inputs():
    age = st.number_input("Age",0,120,52)
    sex = st.selectbox("Sex (0=F,1=M)",[0,1])
    cp = st.number_input("Chest Pain Type",0,3,0)
    trestbps = st.number_input("BP",80,200,120)
    chol = st.number_input("Cholesterol",100,600,240)
    fbs = st.selectbox("FBS > 120",[0,1])
    restecg = st.number_input("Rest ECG",0,2,1)
    thalach = st.number_input("Max HR",60,250,150)
    exang = st.selectbox("Exercise angina",[0,1])
    oldpeak = st.number_input("ST Depression",0.0,10.0,1.2)
    slope = st.number_input("Slope ST",0,2,1)
    ca = st.number_input("Vessels colored",0,3,0)
    thal = st.number_input("Thalassemia",1,3,2)
    return [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

def diabetes_inputs():
    preg = st.number_input("Pregnancies",0,20,2)
    glucose = st.number_input("Glucose",0,300,120)
    bp = st.number_input("BP",0,200,70)
    skin = st.number_input("Skin Thickness",0,100,20)
    insulin = st.number_input("Insulin",0,900,85)
    bmi = st.number_input("BMI",0.0,70.0,28.5)
    dpf = st.number_input("DPF",0.0,3.0,0.5)
    age = st.number_input("Age",1,120,32)
    return [preg,glucose,bp,skin,insulin,bmi,dpf,age]

def kidney_inputs():
    age = st.number_input("Age",0,120,45)
    bp = st.number_input("Blood Pressure",0,200,80)
    sg = st.number_input("Specific Gravity",1.0,1.05,1.02)
    al = st.number_input("Albumin",0,5,0)
    su = st.number_input("Sugar",0,5,0)
    bgr = st.number_input("Blood Glucose Random",0,500,110)
    bu = st.number_input("Blood Urea",0,200,25)
    sc = st.number_input("Serum Creatinine",0.0,20.0,1.0)
    hemo = st.number_input("Hemoglobin",0.0,20.0,15.2)
    pcv = st.number_input("Packed Cell Volume",0,60,44)
    return [age,bp,sg,al,su,bgr,bu,sc,hemo,pcv]

def liver_inputs():
    age = st.number_input("Age",1,120,45)
    gender = st.selectbox("Gender",["Male","Female"])
    gender_val = 1 if gender=="Male" else 0
    total_bilirubin = st.number_input("Total Bilirubin",0.0,10.0,1.3)
    direct_bilirubin = st.number_input("Direct Bilirubin",0.0,5.0,0.4)
    alk_phos = st.number_input("Alkaline Phosphotase",50,2000,210)
    alt = st.number_input("ALT",1,2000,35)
    ast = st.number_input("AST",1,2000,40)
    total_proteins = st.number_input("Total Proteins",1.0,10.0,6.8)
    albumin = st.number_input("Albumin",1.0,6.0,3.1)
    ag_ratio = st.number_input("Albumin/Globulin Ratio",0.0,3.0,0.9)
    return [age,gender_val,total_bilirubin,direct_bilirubin,alk_phos,alt,ast,total_proteins,albumin,ag_ratio]

# ===================== BRAIN TUMOR PREDICTION PAGE =====================
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Brain Tumor Prediction", layout="centered")
st.title("🧠 Brain Tumor Prediction")

# ================= LOAD MODEL =================
@st.cache_resource
def load_brain_model():
    FILE_ID = "1r7Kmf14ZGKQK3GSTk3nxPxfAyGpg2m_b"
    URL = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
    
    response = requests.get(URL)
    with open("brain_tumor_dataset.h5", "wb") as f:
        f.write(response.content)
    
    model = load_model("brain_tumor_dataset.h5")
    return model

model = load_brain_model()

# Show model input shape for debugging
st.write("Model input shape:", model.input_shape)

# ================= IMAGE UPLOAD =================
st.subheader("Upload MRI Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # ================= PREPROCESS IMAGE =================
    # Get model input shape (ignore batch size)
    input_shape = model.input_shape[1:]  # e.g., (86528,) or (128,128,3)
    
    if len(input_shape) == 1:
        # Dense model expecting flattened input
        # Calculate the side of square image if needed
        side = int(np.sqrt(input_shape[0] / 3))
        img = image.resize((side, side))
        img_array = np.array(img) / 255.0
        img_array = img_array.flatten()
        img_array = np.expand_dims(img_array, axis=0)
    else:
        # CNN model expecting (H, W, C)
        img = image.resize((input_shape[0], input_shape[1]))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    # ================= PREDICTION =================
    if st.button("🔍 Predict Brain Tumor"):
        try:
            prediction = model.predict(img_array)[0][0]
            if prediction > 0.5:
                st.error("⚠️ Brain Tumor Detected")
            else:
                st.success("✅ No Brain Tumor Detected")
        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

st.markdown("---")
st.markdown("Made with ❤️ by your ML buddy")

# ===================== APPOINTMENTS =====================
def appointment_booking(disease):
    st.subheader("📅 Doctor Consultation")
    doctor_map = {
        "Heart Disease": ("Cardiologist","https://www.apollo247.com/specialties/cardiology"),
        "Diabetes": ("Diabetologist","https://www.apollo247.com/specialties/diabetology"),
        "Kidney Disease": ("Nephrologist","https://www.apollo247.com/specialties/nephrology"),
        "Liver Disease": ("Hepatologist","https://www.apollo247.com/specialties/hepatology"),
        "Brain Tumor": ("Neurologist","https://www.apollo247.com/specialties/neurology")
    }
    doctor, link = doctor_map.get(disease, ("General Physician","https://www.apollo247.com"))
    st.markdown(f"👨‍⚕️ **Recommended Doctor:** {doctor}")
    st.warning("⚠️ Consult a certified doctor for confirmation")
    st.markdown(f"🔗 **Online Consultation:** [Book Appointment]({link})")

    username = st.session_state['current_user']
    if username not in st.session_state['appointments']:
        st.session_state['appointments'][username] = []
    if st.button("✅ Save Appointment"):
        st.session_state['appointments'][username].append({"disease":disease,"doctor":doctor,"link":link,"time":str(datetime.now())})
        st.success("Appointment added to history!")
    if st.session_state['appointments'].get(username):
        st.subheader("📋 Appointment History")
        for appt in st.session_state['appointments'][username]:
            st.write(f"- **{appt['disease']}** with {appt['doctor']} ➡️ [Link]({appt['link']}) (Saved: {appt['time']})")
def show_hospitals(disease):
    st.subheader("🏥 Nearby Hospitals / Clinics")
    search_map = {
        "Heart Disease": "cardiology hospital near me",
        "Diabetes": "diabetes clinic near me",
        "Kidney Disease": "nephrology hospital near me",
        "Liver Disease": "hepatology hospital near me",
        "Brain Tumor": "neurology hospital near me"
    }
    query = search_map.get(disease, "hospital near me")
    maps_link = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
    st.markdown(f"🔍 **Search Hospitals:** [Click Here]({maps_link})")
# ===================== SPEECH TO TEXT =====================
def speech_to_text_page():
    st.header("🎙️ Speech to Text")
    audio_file = st.file_uploader("Upload WAV file", type=["wav"])
    if audio_file:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_file.read())
            file_path = f.name
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success("Recognized Text:")
            st.text_area("Result", text, height=150)
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error(f"API Error: {e}")

# ===================== MAIN =====================
if st.session_state['page'] == 'Signup':
    signup()
elif st.session_state['page'] == 'Login':
    login()
elif st.session_state['page'] == 'Home':
    home_dashboard()
elif st.session_state['page']=="Heart":
    disease_page("Heart Disease", lambda: load_pickle_model("models/heart_model.pkl"), heart_inputs)
elif st.session_state['page']=="Diabetes":
    disease_page("Diabetes", lambda: load_pickle_model("models/diabetes_model.pkl"), diabetes_inputs)
elif st.session_state['page']=="Kidney":
    disease_page("Kidney Disease", lambda: load_pickle_model("models/kidney_10f_model.pkl"), kidney_inputs)
elif st.session_state['page']=="Liver":
    disease_page("Liver Disease", lambda: load_pickle_model("models/liver_model.pkl"), liver_inputs)
elif st.session_state['page']=="Brain":
    load_brain_model()
elif st.session_state['page']=="Speech":
    speech_to_text_page()
