import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from PIL import Image
import requests
from tensorflow.keras.models import load_model

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Multi Disease Diagnostic System",
    layout="wide"
)

# ==================================================
# SESSION STATE INIT
# ==================================================
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "users" not in st.session_state:
    st.session_state.users = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""

# ==================================================
# GLOBAL CSS
# ==================================================
st.markdown("""
<style>
.stApp { background-color: white; }

.auth-card, .card {
    background: #4ade80;
    border: 2px solid red;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.3);
}

button {
    border-radius: 10px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# PDF CREATOR (SAFE)
# ==================================================
def create_pdf(text, chart_buf=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    safe_text = text.encode("latin1", "ignore").decode("latin1")
    pdf.multi_cell(0, 8, safe_text)

    if chart_buf:
        pdf.add_page()
        pdf.image(chart_buf, x=20, y=30, w=170)

    return pdf.output(dest="S").encode("latin1")

# ==================================================
# RISK ANALYSIS + CHART
# ==================================================
def show_risk(title, probability):
    risk = round(probability * 100, 2)

    st.subheader("📊 Risk Analysis")
    st.metric("Risk Percentage", f"{risk}%")

    fig, ax = plt.subplots()
    ax.bar(["Risk"], [risk])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title(f"{title} Risk Level")

    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return risk, buf

# ==================================================
# RECOMMENDATIONS
# ==================================================
def recommendations(disease, risk):
    if risk < 30:
        return "Low Risk\n• Maintain healthy diet\n• Regular exercise\n• Yearly checkup"
    elif risk < 70:
        return "Medium Risk\n• Reduce sugar & salt\n• Exercise daily\n• Monthly monitoring"
    else:
        return "High Risk\n• Consult doctor immediately\n• Strict diet\n• Avoid alcohol & smoking"

# ==================================================
# MODEL LOADERS
# ==================================================
@st.cache_resource
def load_pickle_model(path):
    with open(path, "rb") as f:
        model, scaler = pickle.load(f)
    return model, scaler

@st.cache_resource
def load_brain_model():
    FILE_ID = "1r7Kmf14ZGKQK3GSTk3nxPxfAyGpg2m_b"
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    r = requests.get(url)
    with open("brain.h5", "wb") as f:
        f.write(r.content)
    return load_model("brain.h5")

# ==================================================
# AUTH PAGES
# ==================================================
def signup():
    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.header("📝 Signup")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Signup"):
        if u == "" or p == "":
            st.error("All fields required")
        elif u in st.session_state.users:
            st.error("User exists")
        else:
            st.session_state.users[u] = p
            st.success("Signup success")
            st.session_state.page = "Login"
    st.markdown("</div>", unsafe_allow_html=True)

def login():
    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.header("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in st.session_state.users and st.session_state.users[u] == p:
            st.session_state.logged_in = True
            st.session_state.current_user = u
            st.session_state.page = "Home"
        else:
            st.error("Invalid credentials")
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# HOME DASHBOARD
# ==================================================
def home():
    st.header("🩺 Multi Disease Diagnostic Portal")
    st.write(f"Welcome **{st.session_state.current_user}**")

    col1, col2, col3 = st.columns(3)

    if col1.button("❤️ Heart"):
        st.session_state.page = "Heart"
    if col2.button("🩸 Diabetes"):
        st.session_state.page = "Diabetes"
    if col3.button("🟣 Kidney"):
        st.session_state.page = "Kidney"

    col4, col5 = st.columns(2)
    if col4.button("🟠 Liver"):
        st.session_state.page = "Liver"
    if col5.button("🧠 Brain Tumor"):
        st.session_state.page = "Brain"

    if st.button("Logout"):
        st.session_state.page = "Login"
        st.session_state.logged_in = False

# ==================================================
# DISEASE PAGE TEMPLATE
# ==================================================
def disease_page(title, model_loader, inputs=None, is_image=False):
    st.header(title)

    if is_image:
        file = st.file_uploader("Upload MRI", type=["jpg","png"])
        if not file:
            return
        img = Image.open(file).resize((224,224))
        st.image(img)
    else:
        data = inputs()

    if st.button("Predict"):
        if is_image:
            model = model_loader()
            X = np.expand_dims(np.array(img)/255, 0)
            prob = model.predict(X)[0][0]
        else:
            model, scaler = model_loader()
            X = scaler.transform([data])
            prob = model.predict_proba(X)[0][1]

        risk, chart = show_risk(title, prob)
        advice = recommendations(title, risk)

        st.subheader("📝 Recommendation")
        st.info(advice)

        report = f"""
User: {st.session_state.current_user}
Disease: {title}
Risk: {risk}%

Advice:
{advice}
"""

        pdf = create_pdf(report, chart)
        st.download_button("📄 Download Report", pdf, f"{title}.pdf", "application/pdf")

    st.button("⬅ Back", on_click=lambda: setattr(st.session_state, "page", "Home"))

# ==================================================
# INPUT FORMS
# ==================================================
def heart_inputs():
    return [st.number_input("Age",0,120,50),
            st.selectbox("Sex",[0,1]),
            st.number_input("Cholesterol",100,600,240)]

def diabetes_inputs():
    return [st.number_input("Glucose",0,300,120),
            st.number_input("BMI",0.0,70.0,25.0),
            st.number_input("Age",1,120,30)]

def kidney_inputs():
    return [st.number_input("BP",0,200,80),
            st.number_input("Creatinine",0.0,20.0,1.0),
            st.number_input("Age",1,120,45)]

def liver_inputs():
    return [st.number_input("Bilirubin",0.0,10.0,1.2),
            st.number_input("ALT",0,2000,35),
            st.number_input("Age",1,120,40)]

# ==================================================
# ROUTER
# ==================================================
if st.session_state.page == "Signup":
    signup()
elif st.session_state.page == "Login":
    login()
elif st.session_state.page == "Home":
    home()
elif st.session_state.page == "Heart":
    disease_page("Heart Disease", lambda: load_pickle_model("models/heart_model.pkl"), heart_inputs)
elif st.session_state.page == "Diabetes":
    disease_page("Diabetes", lambda: load_pickle_model("models/diabetes_model.pkl"), diabetes_inputs)
elif st.session_state.page == "Kidney":
    disease_page("Kidney Disease", lambda: load_pickle_model("models/kidney_model.pkl"), kidney_inputs)
elif st.session_state.page == "Liver":
    disease_page("Liver Disease", lambda: load_pickle_model("models/liver_model.pkl"), liver_inputs)
elif st.session_state.page == "Brain":
    disease_page("Brain Tumor", load_brain_model, is_image=True)
