
# 🩺 Multi-Disease Diagnostic App

A Streamlit web application that uses Machine Learning and Deep Learning to predict multiple diseases from user inputs and medical images.

---

## Features

- 🔐 User Signup & Login
- ❤️ Heart Disease Prediction
- 🩸 Diabetes Prediction
- 🧠 Brain Tumor Detection (MRI Image)
- 🟣 Kidney Disease Prediction
- 🟠 Liver Disease Prediction
- 🎙️ Speech to Text (WAV file)
- 📄 PDF Report Download after each prediction
- 📅 Doctor Appointment Booking (Apollo 247)
- 🏥 Nearby Hospital Search (Google Maps)

---

## Project Structure

```
Multi-Disease-Diagnostic-App/
├── app.py
├── requirements.txt
├── runtime.txt
├── models/
│   ├── heart_model.pkl
│   ├── diabetes_model.pkl
│   ├── kidney_10f_model.pkl
│   ├── liver_model.pkl
│   └── brain_tumor_model.h5
├── pages/
└── .streamlit/
```

---

## Tech Stack

- **Frontend:** Streamlit
- **ML Models:** Scikit-learn (Pickle)
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** Pillow
- **PDF Generation:** FPDF
- **Speech Recognition:** SpeechRecognition

---

## Installation

```bash
git clone https://github.com/Miduna-Varshini/Multi-Disease-Diagnostic-App.git
cd Multi-Disease-Diagnostic-App
pip install -r requirements.txt
streamlit run app.py
```

---

## How to Use

1. Sign up and log in
2. Select a disease from the dashboard
3. Enter the required inputs or upload an MRI image
4. Click **Predict** to get the result
5. Download the PDF report
6. Book a doctor appointment or find nearby hospitals

---

## Disclaimer

> This app is for **educational purposes only** and is not a substitute for professional medical advice. Always consult a qualified doctor.

---

## Author

**Miduna Varshini** — [@Miduna-Varshini](https://github.com/Miduna-Varshini)
