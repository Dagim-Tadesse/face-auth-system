"""
Face Authentication System - Main Application
Streamlit app entry point
"""
import streamlit as st

st.set_page_config(
    page_title="Face Auth System",
    layout="centered"
)

def main():
    st.title("Face Authentication System")
    
    st.markdown("""
    ## Welcome to the Face Authentication System
    Choose an option below to proceed:
    """)
    col1, col2 = st.columns(2)
    
    with col1:
        st.page_link("pages/login.py", label="Login")
    
    with col2:
        st.page_link("pages/register.py", label="Register")


if __name__ == "__main__":
    main()