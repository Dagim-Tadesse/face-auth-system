import streamlit as st
from datetime import datetime

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
    </style>
""", unsafe_allow_html=True)

if "login_status" not in st.session_state:
    st.session_state.login_status = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None

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
        st.image(camera_image, caption="Captured Face", use_column_width=True)

with col2:
    st.markdown("#### Registered Users (Demo)")
    st.caption("For demonstration purposes only")

    demo_users = ["Nati", "Chen", "Darius"]

    for user in demo_users:
        st.markdown(f"👤 **{user}**")

st.divider()

if camera_image is not None:
    if st.button("Authenticate Face", type="primary", use_container_width=True):
        with st.spinner("Analyzing facial features..."):
            # Simulate processing delay
            import time
            time.sleep(2.0)
            import random
            matched_user = random.choice(["Nati", "Chen", "Darius"])
            
            st.session_state.login_status = "success"
            st.session_state.current_user = matched_user
    if st.session_state.login_status == "success":
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #d4edda; border: 1px solid #4CAF50;">
                <h3 class="status-success">Login Successful!</h3>
                <p>Welcome back, <strong>{st.session_state.current_user}</strong></p>
                <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Continue to Dashboard"):
            st.switch_page("pages/dashboard.py")

    elif st.session_state.login_status == "failed":
        st.error("Face not recognized. Please try again or register a new face.")

else:
    st.button("Authenticate Face", type="primary", disabled=True, use_container_width=True)

# Additional options
st.divider()

col_a, col_b = st.columns(2)

with col_a:
    if st.button("Register New Face", use_container_width=True):
        st.info("Registration flow would open here in a full application.")

        
st.caption("Face Login Demo ")