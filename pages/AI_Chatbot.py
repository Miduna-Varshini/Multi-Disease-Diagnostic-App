import streamlit as st
import google.generativeai as genai

# ------------------- Page Config -------------------
st.set_page_config(page_title="AI Health Assistant", page_icon="🤖")

st.title("🤖 AI Health Assistant")
st.write("Ask questions about symptoms, diseases, reports, or prevention.")

# ------------------- Load API Key -------------------
# Make sure you have .streamlit/secrets.toml with:
# GEMINI_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in secrets.toml")
    st.stop()

# ------------------- Configure Gemini -------------------
genai.configure(api_key=api_key)

# Use a valid model
model = genai.GenerativeModel("models/gemini-2.0-flash")

# ------------------- Initialize Chat -------------------
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------- Display Chat History -------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------- User Input -------------------
user_input = st.chat_input("Ask your health question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send message to Gemini
    try:
        response = st.session_state.chat.send_message(user_input)
        reply = response.text
    except Exception as e:
        # Catch any error (quota, 403, 404, etc.)
        reply = f"⚠️ Error: {e}"

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
