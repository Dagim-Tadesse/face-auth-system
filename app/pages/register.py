import os
import sys
from pathlib import Path
import streamlit as st
import re
from datetime import datetime
import cv2
import numpy as np

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CAPTURE_COUNT, user_raw_dir, ensure_project_directories, IMAGE_SIZE
from src.preprocessing import get_preprocessed_face_image

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
        return False, "User ID must be at most 20 characters"
    if not USER_ID_REGEX.match(user_id):
        return False, "User ID can only contain letters, numbers, and underscores"
    return True, ""


def load_face_cascade():
    """Load OpenCV's Haar Cascade for face detection."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)


def detect_face(frame, face_cascade):
    """Detect faces in the frame and return face rectangle coordinates."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


def draw_face_box(frame, faces):
    """Draw bounding boxes around detected faces."""
    output = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if len(faces) == 1:
        cv2.putText(output, "Face Detected - Capturing...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif len(faces) > 1:
        cv2.putText(output, "Multiple Faces - Use Single Face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(output, "Position Face in Camera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output


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
        .progress-container {
            margin: 20px 0;
            padding: 15px;
            background-color: #232424;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        .warning{
            color: #f44336; 
            font-size: 0.9rem;
            }
        .progress-bar-bg {
            height: 25px;
            background-color: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-bar-fill {
            height: 100%;
            background: #037bfc;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .captured-count {
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
            color: #2196F3;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "registration_status" not in st.session_state:
    st.session_state.registration_status = None
if "registered_user" not in st.session_state:
    st.session_state.registered_user = None
if "captured_images" not in st.session_state:
    st.session_state.captured_images = []
if "capture_complete" not in st.session_state:
    st.session_state.capture_complete = False

st.title("Face Registration")
st.markdown("### Register your face for secure authentication")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Step 1: Enter your name")
    user_name = st.text_input(
        label="Full Name",
        placeholder="Enter your full name",
        key="reg_name_input",
        help="Enter the name you want to associate with your face (2-50 characters, letters only)"
    )
    name_valid, name_error = validate_name(user_name)
    if user_name and not name_valid:
        st.markdown(f'<p class="warning">{name_error}</p>', unsafe_allow_html=True)
    
    st.markdown("#### Step 2: Enter your User ID")
    user_id = st.text_input(
        label="User ID",
        placeholder="Enter your User ID",
        key="reg_user_id_input",
        help="Enter the User ID you want to associate with your face (3-20 characters, letters, numbers, underscores only)"
    )
    
    user_id_valid, user_id_error = validate_user_id(user_id)
    if user_id and not user_id_valid:
        st.markdown(f'<p class="warning">{user_id_error}</p>', unsafe_allow_html=True)

with col2:
    st.markdown("#### Registration Tips")
    st.markdown("""
    - Face the camera directly
    - Ensure good lighting
    - Keep a neutral expression
    - Remove glasses if possible
    - Capture 15 images from different angles
    """)
    
    st.divider()
    
    st.markdown("#### Already Registered?")
    if st.button("Go to Login", width="stretch"):
        st.switch_page("pages/login.py")

st.divider()

is_valid = name_valid and user_id_valid

if not is_valid:
    st.warning("Please complete all fields above to start face capture.")

if is_valid:
    save_dir = user_raw_dir(user_id)

# Face capture section
if is_valid and not st.session_state.capture_complete:
    st.markdown("#### Step 3: Capture your face")
    st.info(f"Position your face in the camera. The system will automatically capture {CAPTURE_COUNT} images when your face is detected.")
    
    # Create directory for user
    ensure_project_directories()
    os.makedirs(save_dir, exist_ok=True)

    face_cascade = load_face_cascade()
    
    # Video capture using OpenCV
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access the webcam. Please ensure camera permissions are granted.")
    else:
        # Create containers for video, status, and progress
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Initialize capture state
        last_capture_time = 0
        import time
        
        # Show initial progress
        initial_count = len(st.session_state.captured_images)
        initial_progress = (initial_count / CAPTURE_COUNT) * 100
        progress_placeholder.markdown(f"""
            <div class="progress-container">
                <div class="captured-count">Images Captured: {initial_count} / {CAPTURE_COUNT}</div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {initial_progress}%">
                        {initial_progress:.0f}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        
        # Run capture loop
        while len(st.session_state.captured_images) < CAPTURE_COUNT:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Failed to capture frame from webcam")
                break
            
            # Detect faces
            faces = detect_face(frame, face_cascade)
            
            # Draw face boxes
            display_frame = draw_face_box(frame, faces)
            
            # Convert BGR to RGB for Streamlit display
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Update video display
            video_placeholder.image(display_frame_rgb, channels="RGB", width="stretch")
            
            # Auto-capture if face detected (with debounce)
            current_time = time.time()
            if len(faces) == 1 and (current_time - last_capture_time) > 0.5:
                last_capture_time = current_time
                
                # Use the shared preprocessing function to get the processed face image
                processed_face = get_preprocessed_face_image(frame, target_size=IMAGE_SIZE)
                
                if processed_face is not None:
                    # Save the preprocessed face image
                    filename = save_dir / f"image_of_{user_id}_{len(st.session_state.captured_images) + 1}.jpg"
                    cv2.imwrite(str(filename), processed_face)
                    
                    st.session_state.captured_images.append(str(filename))
                
                # Update progress display
                new_count = len(st.session_state.captured_images)
                new_progress = (new_count / CAPTURE_COUNT) * 100
                
                # Update progress bar
                progress_placeholder.markdown(f"""
                    <div class="progress-container">
                        <div class="captured-count">Images Captured: {new_count} / {CAPTURE_COUNT}</div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: {new_progress}%">
                                {new_progress:.0f}%
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                status_placeholder.success(f"Captured {new_count}/{CAPTURE_COUNT} images!")
                
            elif len(faces) > 1:
                status_placeholder.warning("Multiple faces detected! Please ensure only your face is visible.")
            else:
                status_placeholder.info("👤 Position your face in the camera view...")
        
        cap.release()
        
        # Mark capture as complete
        if len(st.session_state.captured_images) >= CAPTURE_COUNT:
            st.session_state.capture_complete = True
            st.success(f"All {CAPTURE_COUNT} images captured successfully!")

# Registration complete section
if st.session_state.capture_complete:
    st.markdown("---")
    st.markdown("### Face Capture Complete!")
    
    # Show sample of captured images
    st.markdown("#### Captured Images Preview:")
    
    # Display first few images as thumbnails
    cols = st.columns(min(5, len(st.session_state.captured_images)))
    for idx, col in enumerate(cols):
        if idx < len(st.session_state.captured_images):
            with col:
                st.image(st.session_state.captured_images[idx], 
                        caption=f"Image {idx + 1}", 
                        width="stretch")
    
    # Show file locations
    st.markdown(f"""
        <div style="padding: 15px; border-radius: 8px; margin: 15px 0; background-color: #252729;">
            <strong> Files saved to:</strong> <code>{save_dir}</code><br>
            <strong>Total images:</strong> {len(st.session_state.captured_images)}
        </div>
    """, unsafe_allow_html=True)
    
    # Register button
    if st.button("Complete Registration", type="primary", width="stretch"):
        with st.spinner("Processing and registering your face..."):
            import time
            time.sleep(1.0)
            
            st.session_state.registration_status = "success"
            st.session_state.registered_user = user_name
            st.session_state.captured_images = []
            st.session_state.capture_complete = False
            
            # Invalidate login page cache so new user appears immediately
            if "registered_users_cache" in st.session_state:
                del st.session_state["registered_users_cache"]

    if st.button("Retake Photos", width="stretch"):
        # Clean up previously captured images
        if save_dir.exists():
            for f in save_dir.glob("*.jpg"):
                f.unlink()
        st.session_state.captured_images = []
        st.session_state.capture_complete = False
        st.rerun()

# Success state
if st.session_state.registration_status == "success":
    st.markdown(f"""
        <div >
            <h3 class="status-success">Registration Successful!</h3>
            <p>Welcome, <strong>{st.session_state.registered_user}</strong></p>
            <p>Your face has been securely registered with ID: <strong>{user_id}</strong></p>
            <p>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.success("You can now log in using your registered face!")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Continue to Login", type="primary", width="stretch"):
            st.switch_page("pages/login.py")
    with col_btn2:
        if st.button("Register Another Face", width="stretch"):
            st.session_state.registration_status = None
            st.session_state.registered_user = None
            st.session_state.captured_images = []
            st.session_state.capture_complete = False
            st.rerun()

elif st.session_state.registration_status == "failed":
    st.error("Registration failed. Please try again.")
