import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Prediction")
st.caption("Upload MRI images to check for brain tumor")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_brain_model():
    model = load_model("models/brain_tumor_dataset.h5")
    return model

model = load_brain_model()

# ---------------- IMAGE UPLOAD ----------------
st.subheader("Upload MRI Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess image (resize to 128x128, normalize)
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

    if st.button("🔍 Predict Brain Tumor"):
        try:
            pred = model.predict(img_array)[0][0]
            if pred > 0.5:
                st.error("⚠️ Brain Tumor Detected")
            else:
                st.success("✅ No Brain Tumor Detected")
        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

# ---------------- DISCLAIMER ----------------
st.markdown("---")
st.caption("⚠️ For educational purposes only. Consult a doctor for medical advice.")
