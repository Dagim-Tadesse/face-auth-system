import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

try:
    from config import (
        CAPTURE_COUNT,
        LABEL_ENCODER_PATH,
        MODEL_PATH,
        ensure_project_directories,
        user_raw_dir,
    )
    from src.feature_engineering import all_images
    from src.preprocessing import _detect_valid_faces, assess_face_quality
    from src.train import train_model
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from config import (
        CAPTURE_COUNT,
        LABEL_ENCODER_PATH,
        MODEL_PATH,
        ensure_project_directories,
        user_raw_dir,
    )
    from src.feature_engineering import all_images
    from src.preprocessing import _detect_valid_faces, assess_face_quality
    from src.train import train_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


NAME_REGEX_HINT = "Use a short user name or ID."


def detect_face(frame):
    return _detect_valid_faces(frame, to_gray=True)


def draw_face_box(frame, faces):
    output = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output


def audit_saved_images_quality(save_dir):
    image_paths = sorted(save_dir.glob("*.jpg")) + \
        sorted(save_dir.glob("*.jpeg"))
    if len(image_paths) == 0:
        return 0, 0, []

    good_count = 0
    failed = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        is_good, reasons, _ = assess_face_quality(image)
        if is_good:
            good_count += 1
        else:
            failed.append((image_path.name, reasons))

    return len(image_paths), good_count, failed


def warning_box(message):
    st.markdown(
        f"""
        <div class="warning-box">
            <strong>Warning:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Face Registration",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=DM+Sans:wght@400;500;700&display=swap');

        :root {
            --ink: #12263f;
            --muted: #415a77;
            --card: rgba(255,255,255,.86);
            --stroke: rgba(18, 38, 63, .18);
            --accent: #0f766e;
            --accent-2: #c2410c;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 14%, rgba(15, 118, 110, .13), transparent 39%),
                radial-gradient(circle at 90% 12%, rgba(194, 65, 12, .15), transparent 35%),
                linear-gradient(180deg, #f2efe5 0%, #fbfbfb 68%);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        .main .block-container {
            max-width: 1080px;
            padding-top: 1rem;
        }

        .hero-register {
            border-radius: 20px;
            border: 1px solid var(--stroke);
            background: linear-gradient(128deg, rgba(255,255,255,.91), rgba(255,255,255,.72));
            box-shadow: 0 14px 34px rgba(18, 38, 63, 0.09);
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        }

        .hero-register h1 {
            margin: 0;
            font-family: 'Sora', sans-serif;
            font-weight: 800;
            font-size: clamp(1.4rem, 3.3vw, 2.1rem);
        }

        .hero-register p {
            margin: .45rem 0 0 0;
            color: var(--muted);
            font-size: .97rem;
        }

        .kicker {
            display: inline-block;
            font-size: .74rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--accent);
            font-weight: 700;
            background: rgba(15, 118, 110, .12);
            border-radius: 999px;
            padding: .2rem .58rem;
            margin-bottom: .45rem;
        }

        .panel-card {
            border-radius: 14px;
            border: 1px solid var(--stroke);
            background: var(--card);
            box-shadow: 0 10px 24px rgba(18, 38, 63, .08);
            padding: .8rem .9rem;
            margin-bottom: .8rem;
        }

        .stButton>button {
            width: 100%;
            min-height: 2.85rem;
            font-size: 1.02rem;
            border-radius: 12px;
            border: 1px solid rgba(18, 38, 63, .22);
            background: rgba(255, 255, 255, 0.95);
            color: #0f172a;
            font-weight: 700;
            transition: transform .12s ease, box-shadow .12s ease;
        }

        .stButton>button[kind="primary"] {
            border: 1px solid rgba(13, 148, 136, .58);
            background: linear-gradient(135deg, #0f766e, #0d9488);
            color: #f8fafc;
            box-shadow: 0 10px 22px rgba(15, 118, 110, .24);
        }

        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(15, 118, 110, .16);
        }

        .stButton>button:disabled {
            background: #dbe3ea;
            color: #6b7280;
            border-color: #cbd5e1;
            box-shadow: none;
        }

        .progress-box {
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(15, 118, 110, .24);
            background: rgba(15, 118, 110, .09);
            color: var(--ink);
            font-weight: 600;
        }

        .warning-box {
            padding: 0.95rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(245, 158, 11, 0.35);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.94));
            color: #f8fafc;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.22);
            font-weight: 600;
            margin: 0.65rem 0;
        }

        .warning-box strong {
            color: #f8fafc;
        }

        @media (max-width: 768px) {
            .hero-register { padding: 1rem; }
            .stButton>button { min-height: 2.65rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "capture_count" not in st.session_state:
    st.session_state.capture_count = 0
if "saved_user" not in st.session_state:
    st.session_state.saved_user = ""
if "capture_started" not in st.session_state:
    st.session_state.capture_started = False
if "capture_done" not in st.session_state:
    st.session_state.capture_done = False
if "quality_warning" not in st.session_state:
    st.session_state.quality_warning = ""
if "active_name" not in st.session_state:
    st.session_state.active_name = ""

st.markdown(
    """
    <section class="hero-register">
        <span class="kicker">Enrollment</span>
        <h1>Face Registration</h1>
        <p>Capture clean, varied face images to improve authentication quality and reduce false rejects.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.caption(NAME_REGEX_HINT)

name = st.text_input("Enter your name or ID", placeholder="e.g. dragondagi")

if name:
    if st.session_state.active_name != name:
        st.session_state.active_name = name
        st.session_state.capture_count = 0
        st.session_state.capture_done = False
        st.session_state.capture_started = False
        st.session_state.quality_warning = ""

    save_dir = user_raw_dir(name)
    existing_images = sorted(save_dir.glob("*.jpg")) + \
        sorted(save_dir.glob("*.jpeg"))
    username_exists = len(existing_images) > 0
    ensure_project_directories()
    os.makedirs(save_dir, exist_ok=True)
    st.session_state.saved_user = name
    st.caption(f"Images will be saved to: {save_dir}")

    if username_exists:
        warning_box(
            "This username already exists. Please choose another username or delete the existing folder before registering again."
        )

    st.markdown(
        """
        <section class="panel-card">
            <span class="kicker">Capture Controls</span>
            <div style="font-weight:700; margin-top:.1rem;">Webcam Enrollment Panel</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        start_clicked = st.button(
            "Start Auto Capture",
            type="primary",
            disabled=st.session_state.capture_started or username_exists,
        )
    with col_b:
        reset_clicked = st.button("Reset Counter")

    if reset_clicked:
        st.session_state.capture_count = 0
        st.session_state.capture_done = False
        st.session_state.capture_started = False
        st.rerun()

    if start_clicked:
        st.session_state.capture_started = True
        st.session_state.capture_done = False

    progress_placeholder = st.empty()
    camera_col, feedback_col = st.columns([2.15, 1])
    with camera_col:
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    with feedback_col:
        st.markdown(
            """
            <section class="panel-card">
                <span class="kicker">Live Quality</span>
                <div style="font-weight:700; margin-top:.1rem;">Blur and Lighting Feedback</div>
                <div style="color:#415a77; font-size:.92rem; margin-top:.35rem;">Messages will appear here while the camera is running.</div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        warning_placeholder = st.empty()

    progress_placeholder.markdown(
        f"""
        <div class="progress-box">
            <div><strong>Captured:</strong> {st.session_state.capture_count} / {CAPTURE_COUNT}</div>
            <div><strong>Last user:</strong> {st.session_state.saved_user or 'None'}</div>
            <div><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if username_exists:
        st.info(
            "Registration is locked for this name because saved images already exist for it.")
    elif st.session_state.capture_started and not st.session_state.capture_done:
        if st.session_state.capture_count >= CAPTURE_COUNT:
            st.session_state.capture_count = 0

        camera = cv2.VideoCapture(0)
        last_saved_time = 0.0
        start_count = st.session_state.capture_count

        if not camera.isOpened():
            st.error(
                "Could not access the webcam. Please ensure camera permissions are granted.")
            st.session_state.capture_started = False
        else:
            while st.session_state.capture_count < CAPTURE_COUNT:
                ret, frame = camera.read()
                if not ret:
                    status_placeholder.error("Could not read from webcam.")
                    break

                faces = detect_face(frame)
                preview = draw_face_box(frame, faces)
                video_placeholder.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    caption="Webcam preview",
                    use_container_width=True,
                )

                now = time.time()
                if len(faces) > 0 and (now - last_saved_time) > 0.7:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    is_good, reasons, metrics = assess_face_quality(
                        frame, face_bbox=largest_face)
                    if is_good:
                        next_index = st.session_state.capture_count + 1
                        filename = save_dir / \
                            f"image_of_{name}_{next_index}.jpg"
                        cv2.imwrite(str(filename), frame)
                        st.session_state.capture_count = next_index
                        last_saved_time = now
                        st.session_state.quality_warning = ""
                        status_placeholder.success(
                            f"Saved image {next_index} of {CAPTURE_COUNT} for {name}."
                        )
                        warning_placeholder.empty()

                        progress_placeholder.markdown(
                            f"""
                            <div class="progress-box">
                                <div><strong>Captured:</strong> {st.session_state.capture_count} / {CAPTURE_COUNT}</div>
                                <div><strong>Last user:</strong> {st.session_state.saved_user or 'None'}</div>
                                <div><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        warning_text = " ".join(reasons)
                        st.session_state.quality_warning = warning_text
                        warning_placeholder.markdown(
                            f"""
                            <div class="warning-box">
                                <strong>Warning:</strong> Skipped frame: {warning_text}
                                <div style="margin-top:0.35rem; font-size:0.92rem; color:#e2e8f0;">
                                    sharpness={metrics.get('sharpness', 0.0):.1f} | brightness={metrics.get('brightness', 0.0):.1f} | contrast={metrics.get('contrast', 0.0):.1f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                time.sleep(0.05)

                if st.session_state.capture_count >= CAPTURE_COUNT:
                    break

            camera.release()
            st.session_state.capture_started = False
            st.session_state.capture_done = True
            saved_this_run = st.session_state.capture_count - start_count
            if st.session_state.capture_count >= CAPTURE_COUNT and saved_this_run > 0:
                st.success(
                    f"Capture complete. Saved {saved_this_run} images for {name}.")
            elif saved_this_run == 0:
                st.info(
                    "No images were saved in this run. Keep your face centered with good lighting and try again.")

    action_a, action_b, action_c = st.columns(3)
    with action_a:
        goto_login = st.button("Go to Login", type="primary")
    with action_b:
        train_clicked = st.button("Train Model")
    with action_c:
        quality_check_clicked = st.button("Check Quality")

    if goto_login:
        st.switch_page("pages/login.py")

    if train_clicked:
        X, _ = all_images()
        if len(X) == 0:
            st.error("No training images found. Capture at least one face first.")
        else:
            with st.spinner("Training model from current registered faces..."):
                train_model()

            if MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
                st.success(
                    "Training complete. You can now go to Login and authenticate.")
            else:
                st.error("Training did not produce model files. Please try again.")

    if quality_check_clicked:
        total, good_count, failed = audit_saved_images_quality(save_dir)
        if total == 0:
            warning_box("No saved images found for this user yet.")
        else:
            st.info(f"Good images: {good_count} / {total}")
            if len(failed) > 0:
                warning_box(
                    "Some images are low quality and should be recaptured:")
                for file_name, reasons in failed[:10]:
                    st.write(f"- {file_name}: {' '.join(reasons)}")
                if len(failed) > 10:
                    st.write(f"...and {len(failed) - 10} more.")
            else:
                st.success("All saved images pass the quality checks.")
else:
    st.info("Enter a name or ID to begin capturing images.")

st.caption("Face Authentication System")
