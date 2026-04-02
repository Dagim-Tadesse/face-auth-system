import os
import sys
from pathlib import Path
import streamlit as st
from datetime import datetime
import cv2
import numpy as np
from collections import defaultdict

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DIR, IMAGE_SIZE
from src.preprocessing import preprocess_frame


st.set_page_config(
    page_title="Face Login",
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
            border: 4px solid #4CAF50;
            border-radius: 15px;
            padding: 10px;
            background-color: #f8f9fa;
        }
        .status-success {
            color: #4CAF50;
            font-weight: bold;
        }
        .status-error {
            color: #f44336;
            font-weight: bold;
        }
        .user-card {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border-left: 1px solid #375275;
        }
    </style>
""", unsafe_allow_html=True)

if "login_status" not in st.session_state:
    st.session_state.login_status = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None


def get_registered_users(use_cache=True):
    """Get list of registered users from the data/raw directory."""
    # Use session state caching to avoid redundant filesystem reads
    cache_key = "registered_users_cache"
    if use_cache and cache_key in st.session_state:
        return st.session_state[cache_key]
    
    if not RAW_DIR.exists():
        return []
    
    users = []
    for user_dir in RAW_DIR.iterdir():
        if user_dir.is_dir():
            # Count images in the user directory
            image_count = len(list(user_dir.glob("*.jpg"))) + len(list(user_dir.glob("*.png")))
            users.append({
                "username": user_dir.name,
                "image_count": image_count
            })
    
    # Cache the result
    if use_cache:
        st.session_state[cache_key] = users
    
    return users


def invalidate_user_cache():
    """Clear the cached user list (call after registration or user changes)."""
    if "registered_users_cache" in st.session_state:
        del st.session_state["registered_users_cache"]

# Header
st.title("Face Login")
st.markdown("### Secure Authentication using Facial Recognition")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Step 1: Position your face in the frame")
    st.info("Ensure good lighting and look directly at the camera.")
    camera_image = st.camera_input(
        label="Capture your face for login",
        key="face_camera",
        help="Click 'Take Photo' once your face is clearly visible in the center."
    )

    if camera_image is not None:
        st.success("Photo captured successfully!")
        st.image(camera_image, caption="Captured Face", width="stretch")

with col2:
    st.markdown("#### Registered Users (Demo)")
    st.caption("For demonstration purposes only")

    registered_users = get_registered_users()
    
    if registered_users:
        for user_info in registered_users:
            st.markdown(f"""
                <div class="user-card">
                    <strong>{user_info['username']}</strong>
                    <br><small style="color: #666;">{user_info['image_count']} images</small>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No registered users yet. Register a new face to get started!")

st.divider()

if camera_image is not None:
    if st.button("Authenticate Face", type="primary", width="stretch"):
        with st.spinner("Analyzing facial features..."):
            registered_users = get_registered_users()
            
            if not registered_users:
                st.error("No registered users found. Please register first!")
                st.session_state.login_status = "failed"
            else:
                # Convert camera input bytes to numpy array for processing
                bytes_data = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
                
                # Preprocess the captured frame for face recognition
                processed_features = preprocess_frame(frame, target_size=IMAGE_SIZE)
                
                if processed_features is None:
                    st.error("No face detected in the captured image. Please try again.")
                    st.session_state.login_status = "failed"
                else:
                    """
                    User matching goes here
                    """
                    st.session_state.login_status = "success"
                    st.session_state.current_user = "Nati"
                    st.session_state.match_score = 0.95
    
    if st.session_state.login_status == "success":
        match_score_pct = st.session_state.get('match_score', 0) * 100
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #333a42; border: 1px solid #005d9c;">
                <h3 class="status-success">Login Successful!</h3>
                <p>Welcome back, <strong>{st.session_state.current_user}</strong></p>
                <p>Match confidence: <strong>{match_score_pct:.1f}%</strong></p>
                <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Goto Home page", width="stretch"):
            st.switch_page("main.py")

    elif st.session_state.login_status == "failed":
        st.error("Face not recognized. Please try again or register a new face.")

else:
    st.button("Authenticate Face", type="primary", disabled=True, width="stretch")

# Additional options
st.divider()

col_a, col_b = st.columns(2)

with col_a:
    if st.button("Register New Face", width="stretch"):
        st.switch_page("pages/register.py")

# Refresh button to update user list and invalidate cache
if st.button("Refresh Users", width="stretch"):
    invalidate_user_cache()
    st.rerun()
        
st.caption("Face Authentication System")