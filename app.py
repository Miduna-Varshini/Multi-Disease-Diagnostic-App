import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multi-Disease Diagnostic App",
    page_icon="🩺",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Heart", "Kidney", "Liver", "Diabetes", "Brain Tumor"])

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("🩺 Multi-Disease Diagnostic Portal")
    st.markdown("""
    Welcome! This portal helps you **predict multiple diseases** using AI.
    Select a disease from the sidebar to start.
    """)
    
    # Display boxes for each disease
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            '<div style="background-color:#FFC0CB; border-radius:10px; padding:20px; text-align:center;">'
            '<h3>❤️ Heart</h3>'
            '<p>Check your heart health</p>'
            '</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            '<div style="background-color:#ADD8E6; border-radius:10px; padding:20px; text-align:center;">'
            '<h3>🩸 Diabetes</h3>'
            '<p>Monitor glucose & risk</p>'
            '</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(
            '<div style="background-color:#90EE90; border-radius:10px; padding:20px; text-align:center;">'
            '<h3>🧠 Brain</h3>'
            '<p>Detect brain tumor</p>'
            '</div>', unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown(
            '<div style="background-color:#FFD580; border-radius:10px; padding:20px; text-align:center;">'
            '<h3>🟣 Kidney</h3>'
            '<p>Check kidney function</p>'
            '</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown(
            '<div style="background-color:#FFA07A; border-radius:10px; padding:20px; text-align:center;">'
            '<h3>🟠 Liver</h3>'
            '<p>Check liver health</p>'
            '</div>', unsafe_allow_html=True)

# ---------------- HEART PAGE ----------------
elif page == "Heart":
    # Import Heart page code here
    import pages.1_Heart

# ---------------- KIDNEY PAGE ----------------
elif page == "Kidney":
    import pages.2_Kidney

# ---------------- LIVER PAGE ----------------
elif page == "Liver":
    import pages.3_Liver

# ---------------- DIABETES PAGE ----------------
elif page == "Diabetes":
    import pages.4_Diabetes

# ---------------- BRAIN TUMOR PAGE ----------------
elif page == "Brain Tumor":
    import pages.5_Brain_Tumor

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Made with ❤️ by your ML buddy")
