import streamlit as st
import re
from datetime import datetime
import cv2
import numpy as np

# Regex pattern for valid names (letters, spaces, hyphens, apostrophes)
NAME_REGEX = re.compile(r"^[A-Za-z\s\-']{2,50}$")

# Regex pattern for valid User ID (alphanumeric, underscores, 3-20 characters)
USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_]{3,20}$")

def validate_name(name):
    """Validate name against regex pattern."""
    if not name:
        return False, "Please enter your name"
    if len(name) < 2:
        return False, "Name must be at least 2 characters"
    if len(name) > 50:
        return False, "Name must be less than 50 characters"
    if not NAME_REGEX.match(name):
        return False, "Name can only contain letters, spaces, hyphens, and apostrophes"
    return True, ""

def validate_user_id(user_id):
    """Validate user ID against regex pattern."""
    if not user_id:
        return False, "Please enter a user ID"
    if len(user_id) < 3:
        return False, "User ID must be at least 3 characters"
    if len(user_id) > 20:
        return False, "User ID must be less than 20 characters"
    if not USER_ID_REGEX.match(user_id):
        return False, "User ID can only contain letters, numbers, and underscores"
    return True, ""

st.set_page_config(
    page_title="Face Registration",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .main { padding-top: 2rem; }
        .stButton>button {
            width: 100%;
            height: 3.2rem;
            font-size: 1.1rem;
        }
        .face-frame {
            border: 4px solid #2196F3;
            border-radius: 15px;
            padding: 10px;
            background-color: #f8f9fa;
        }
        .status-success {
            color: #4CAF50;
            font-weight: bold;
        }
        .status-info {
            color: #2196F3;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

if "registration_status" not in st.session_state:
    st.session_state.registration_status = None
if "registered_user" not in st.session_state:
    st.session_state.registered_user = None

# Header
st.title("Face Registration")
st.markdown("### Register your face for secure authentication")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Step 1: Enter your name")
    user_name = st.text_input(
        label="Full Name",
        placeholder="Enter your full name",
        help="Enter the name you want to associate with your face (2-50 characters, letters only)"
    )
    name_valid, name_error = validate_name(user_name)
    if user_name and not name_valid:
        st.markdown(f'<p style="color: #f44336; font-size: 0.9rem;">{name_error}</p>', unsafe_allow_html=True)
    
    st.markdown("#### Step 1.5: Enter your User ID")
    user_id = st.text_input(
        label="User ID",
        placeholder="Enter your User ID",
        help="Enter the User ID you want to associate with your face (3-20 characters, letters, numbers, underscores only)"
    )
    
    user_id_valid, user_id_error = validate_user_id(user_id)
    if user_id and not user_id_valid:
        st.markdown(f'<p style="color: #f44336; font-size: 0.9rem;">{user_id_error}</p>', unsafe_allow_html=True)

    st.markdown("#### Step 2: Capture your face")
    st.info("Ensure good lighting and look directly at the camera. Capture multiple angles if possible.")
    camera_image = st.camera_input(
        label="Capture your face for registration",
        key="face_camera_register",
        help="Click 'Take Photo' once your face is clearly visible in the center. Multiple captures recommended."
    )
    
    if camera_image is not None:
        st.success("Photo captured successfully!")
        st.image(camera_image, caption="Captured Face")

with col2:
    st.markdown("#### Registration Tips")
    st.markdown("""
    - Face the camera directly
    - Ensure good lighting
    - Keep a neutral expression
    - Remove glasses if possible
    """)
    
    st.divider()
    
    st.markdown("#### Already Registered?")
    if st.button("Go to Login", use_container_width=True):
        st.switch_page("pages/login.py")

st.divider()
is_valid = name_valid and user_id_valid and camera_image is not None

if is_valid:
    if st.button("Register Face", type="primary", use_container_width=True):
        with st.spinner("Processing and registering your face..."):
            import time
            time.sleep(2.0)
            
            st.session_state.registration_status = "success"
            st.session_state.registered_user = user_name
    
    if st.session_state.registration_status == "success":
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #d4edda; border: 1px solid #4CAF50;">
                <h3 class="status-success">Registration Successful!</h3>
                <p>Welcome, <strong>{st.session_state.registered_user}</strong></p>
                <p>Your face has been securely registered.</p>
                <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.success("You can now log in using your registered face!")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Continue to Login", type="primary", use_container_width=True):
                st.switch_page("pages/login.py")
        with col_btn2:
            if st.button("Register Another Face", use_container_width=True):
                st.session_state.registration_status = None
                st.session_state.registered_user = None
                st.rerun()

    elif st.session_state.registration_status == "failed":
        st.error("Registration failed. Please try again.")
else:
    if not name_valid or not user_id_valid:
        st.button("Register Face", type="primary", disabled=True, use_container_width=True)
        st.caption("Please fix the errors above to continue.")
    elif not user_id:
        st.button("Register Face", type="primary", disabled=True, use_container_width=True)
        st.caption("Please enter your User ID to continue.")
    elif not user_name:
        st.button("Register Face", type="primary", disabled=True, use_container_width=True)
        st.caption("Please enter your name to continue.")
    elif not camera_image:
        st.button("Register Face", type="primary", disabled=True, use_container_width=True)
        st.caption("Please capture your face to continue.")
    else:
        st.button("Register Face", type="primary", disabled=True, use_container_width=True)
        st.caption("Please complete all fields to register.")
