import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

try:
    from config import RAW_DIR
    from src.predict import FacePredictor
    from src.preprocessing import _detect_valid_faces
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from config import RAW_DIR
    from src.predict import FacePredictor
    from src.preprocessing import _detect_valid_faces

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FaceVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame


def get_registered_users():
    if not RAW_DIR.exists():
        return []
    return [d.name for d in RAW_DIR.iterdir() if d.is_dir()]


def detect_face(frame):
    return _detect_valid_faces(frame, to_gray=True)


def draw_face_box(frame, faces):
    output = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output


def get_predictor():
    if "predictor" not in st.session_state:
        st.session_state.predictor = FacePredictor()
    return st.session_state.predictor


st.set_page_config(page_title="Face Login", layout="centered",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=DM+Sans:wght@400;500;700&display=swap');

        :root {
            --ink: #112240;
            --muted: #3b536c;
            --card: rgba(255, 255, 255, 0.86);
            --stroke: rgba(17, 34, 64, 0.18);
            --accent: #0f766e;
            --accent-2: #d97706;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(15, 118, 110, 0.14), transparent 40%),
                radial-gradient(circle at 88% 14%, rgba(217, 119, 6, 0.17), transparent 38%),
                linear-gradient(180deg, #f4f1e8 0%, #fbfbfa 65%);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        .main .block-container {
            max-width: 1080px;
            padding-top: 1rem;
            animation: fadeRise .35s ease-out;
        }

        @keyframes fadeRise {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .hero-login {
            border-radius: 20px;
            border: 1px solid var(--stroke);
            background: linear-gradient(125deg, rgba(255,255,255,.92), rgba(255,255,255,.72));
            box-shadow: 0 14px 34px rgba(17, 34, 64, 0.09);
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        }

        .hero-login h1 {
            margin: 0;
            font-family: 'Sora', sans-serif;
            font-size: clamp(1.45rem, 3.3vw, 2.1rem);
            font-weight: 800;
            color: var(--ink);
        }

        .hero-login p {
            margin: .45rem 0 0 0;
            color: var(--muted);
            font-size: .98rem;
        }

        .panel-card {
            border-radius: 16px;
            border: 1px solid var(--stroke);
            background: var(--card);
            box-shadow: 0 10px 24px rgba(17, 34, 64, 0.08);
            padding: 0.85rem 0.95rem;
            margin-bottom: .75rem;
        }

        .kicker {
            display: inline-block;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--accent);
            font-weight: 700;
            background: rgba(15, 118, 110, .12);
            border-radius: 999px;
            padding: .2rem .58rem;
            margin-bottom: .45rem;
        }

        .stButton>button {
            width: 100%;
            min-height: 2.85rem;
            font-size: 1.02rem;
            border-radius: 12px;
            border: 1px solid rgba(17, 34, 64, 0.22);
            background: rgba(255, 255, 255, 0.95);
            color: #0f172a;
            font-weight: 700;
            transition: transform .12s ease, box-shadow .12s ease;
        }

        .stButton>button[kind="primary"] {
            border: 1px solid rgba(13, 148, 136, 0.62);
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

        .bottom-nav {
            margin-top: .2rem;
        }

        .search-box {
            padding: 12px;
            border-radius: 12px;
            border: 1px solid rgba(15, 118, 110, 0.28);
            background: rgba(15, 118, 110, 0.1);
            color: var(--ink);
            margin-top: 0.75rem;
            font-weight: 600;
        }

        .user-chip {
            padding: .62rem .7rem;
            margin: .4rem 0;
            border-radius: 10px;
            border: 1px solid rgba(17, 34, 64, 0.16);
            background: rgba(255,255,255,.76);
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
            .hero-login { padding: 1rem; }
            .stButton>button { min-height: 2.65rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "login_status" not in st.session_state:
    st.session_state.login_status = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "match_score" not in st.session_state:
    st.session_state.match_score = 0.0
if "match_distance" not in st.session_state:
    st.session_state.match_distance = 0.0
if "prediction_message" not in st.session_state:
    st.session_state.prediction_message = ""
if "scanning" not in st.session_state:
    st.session_state.scanning = False
if "balloons_shown" not in st.session_state:
    st.session_state.balloons_shown = False
if "video_transformer" not in st.session_state:
    st.session_state.video_transformer = None
if "scan_last_status" not in st.session_state:
    st.session_state.scan_last_status = ""
if "scan_started_at" not in st.session_state:
    st.session_state.scan_started_at = None
if "cooldown_until" not in st.session_state:
    st.session_state.cooldown_until = 0.0

AUTH_TIMEOUT_SECONDS = 90
COOLDOWN_SECONDS = 180
now_ts = time.time()
in_cooldown = now_ts < float(st.session_state.cooldown_until)


def warning_box(message):
    st.markdown(
        f"""
        <div class="warning-box">
            <strong>Warning:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <section class="hero-login">
        <span class="kicker">Authentication</span>
        <h1>Face Login</h1>
        <p>Start live scanning and grant access instantly when a confident match is detected.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        <section class="panel-card">
            <span class="kicker">Live Camera</span>
            <div style="font-weight:700; margin-top:.1rem;">Authentication Stream</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Click Start Face Auth to keep scanning from webcam. "
        "It will continue until a match is found or you press Stop."
    )

    webrtc_ctx = webrtc_streamer(
        key="login-camera",
        video_processor_factory=FaceVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.session_state.video_transformer = webrtc_ctx.video_processor

    c1, c2, c3 = st.columns(3)
    with c1:
        start_clicked = st.button(
            "Start Face Auth", type="primary", disabled=st.session_state.scanning or in_cooldown)
    with c2:
        stop_clicked = st.button(
            "Stop", disabled=not st.session_state.scanning)
    with c3:
        reset_clicked = st.button("Reset Result")

    if in_cooldown:
        remaining_seconds = int(st.session_state.cooldown_until - now_ts)
        mins = remaining_seconds // 60
        secs = remaining_seconds % 60
        warning_box(
            f"Authentication cooldown active. Try again in {mins:02d}:{secs:02d}."
        )

    if start_clicked:
        st.session_state.login_status = None
        st.session_state.current_user = None
        st.session_state.match_score = 0.0
        st.session_state.match_distance = 0.0
        st.session_state.prediction_message = ""
        st.session_state.scan_last_status = "Starting camera scan..."
        st.session_state.balloons_shown = False
        st.session_state.scanning = True
        st.session_state.scan_started_at = time.time()
        st.rerun()

    if stop_clicked:
        st.session_state.scanning = False
        st.session_state.scan_last_status = "Stopped by user."
        st.session_state.scan_started_at = None
        st.info("Face scanning stopped.")

    if reset_clicked:
        st.session_state.login_status = None
        st.session_state.current_user = None
        st.session_state.match_score = 0.0
        st.session_state.match_distance = 0.0
        st.session_state.prediction_message = ""
        st.session_state.scan_last_status = ""
        st.session_state.scanning = False
        st.session_state.balloons_shown = False
        st.session_state.scan_started_at = None
        st.rerun()

    status_placeholder = st.empty()
    preview_placeholder = st.empty()

    if st.session_state.scanning:
        elapsed_total = time.time() - \
            float(st.session_state.scan_started_at or time.time())
        if elapsed_total >= AUTH_TIMEOUT_SECONDS:
            st.session_state.scanning = False
            st.session_state.login_status = "failed"
            st.session_state.prediction_message = "failed ,try again after 3 min"
            st.session_state.scan_last_status = "Authentication timeout reached."
            st.session_state.cooldown_until = time.time() + COOLDOWN_SECONDS
            st.session_state.scan_started_at = None
            st.toast("failed ,try again after 3 min", icon="⚠️")
            st.rerun()

        status_placeholder.markdown(
            """
            <div class="search-box">
                Scanning live camera for a registered face...
            </div>
            """,
            unsafe_allow_html=True,
        )

        predictor = get_predictor()
        found_match = False
        scan_window_seconds = 3.0
        start_time = time.time()

        while time.time() - start_time < scan_window_seconds:
            transformer = st.session_state.get("video_transformer")
            frame = getattr(transformer, "frame",
                            None) if transformer is not None else None
            if frame is None:
                st.session_state.scan_last_status = "Waiting for camera frames..."
                time.sleep(0.1)
                continue

            faces = detect_face(frame)
            preview_placeholder.image(
                cv2.cvtColor(draw_face_box(frame, faces), cv2.COLOR_BGR2RGB),
                caption="Live scan",
                use_container_width=True,
            )

            if len(faces) == 0:
                st.session_state.scan_last_status = "No clear face detected yet."
                time.sleep(0.1)
                continue

            result = predictor.predict_from_frame(frame)
            if result["success"] and result["accepted"]:
                st.session_state.match_score = result["confidence"]
                st.session_state.match_distance = result.get("distance", 0.0)
                st.session_state.current_user = result["predicted_user"]
                st.session_state.prediction_message = result["message"]
                st.session_state.login_status = "success"
                st.session_state.scanning = False
                st.session_state.scan_started_at = None
                st.session_state.scan_last_status = "Match found."
                st.session_state.balloons_shown = False
                found_match = True
                break

            st.session_state.login_status = "failed"
            st.session_state.prediction_message = result.get(
                "message", "Face not recognized.")
            st.session_state.scan_last_status = "Face seen, but no accepted match yet."
            time.sleep(0.1)

        if found_match:
            st.rerun()

        if st.session_state.scanning:
            status_placeholder.markdown(
                f"""
                <div class="search-box">
                    {st.session_state.scan_last_status or 'Still scanning...'}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.rerun()
    elif st.session_state.scan_last_status:
        status_placeholder.markdown(
            f"""
            <div class="search-box">
                {st.session_state.scan_last_status}
            </div>
            """,
            unsafe_allow_html=True,
        )

with col2:
    st.markdown(
        """
        <section class="panel-card">
            <span class="kicker">Directory</span>
            <div style="font-weight:700; margin-top:.1rem;">Registered Users</div>
            <div style="color:#3b536c; font-size:.9rem;">Faces saved in data/raw</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    users = get_registered_users()
    if users:
        for user in users:
            st.markdown(
                f"""
                <div class="user-chip">
                    {user}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No registered users yet. Register a new face to get started!")

st.divider()

if st.session_state.login_status == "success" and not st.session_state.balloons_shown:
    st.balloons()
    st.session_state.balloons_shown = True
    st.success(
        f"Access Granted: {st.session_state.current_user} ({st.session_state.match_score * 100:.1f}%)")
    st.caption(f"Distance: {st.session_state.match_distance:.2f}")
    st.caption(f"Matched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
elif st.session_state.login_status == "success":
    st.success(
        f"Access Granted: {st.session_state.current_user} ({st.session_state.match_score * 100:.1f}%)")
    st.caption(f"Distance: {st.session_state.match_distance:.2f}")
    st.caption(f"Matched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
elif st.session_state.login_status == "failed":
    st.error(
        st.session_state.prediction_message or "Face not recognized. Please try again.")

st.divider()

_, nav_col, _ = st.columns([1.2, 1, 1.2])
with nav_col:
    st.markdown('<div class="bottom-nav">', unsafe_allow_html=True)
    if st.button("Go to Register", type="primary"):
        st.switch_page("pages/register.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Face Authentication System")
