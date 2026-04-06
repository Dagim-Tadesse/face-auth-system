"""
Face Authentication System - Main Application
Streamlit app entry point
"""
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

st.set_page_config(page_title="Face Auth System", layout="wide")


def _registered_user_count() -> int:
    if not RAW_DIR.exists():
        return 0
    return len([d for d in RAW_DIR.iterdir() if d.is_dir()])


def main():
    user_count = _registered_user_count()

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@400;500;700&display=swap');

            :root {
                --bg-top: #f1efe7;
                --bg-bottom: #ffffff;
                --ink: #102a43;
                --muted: #334e68;
                --card: rgba(255, 255, 255, 0.82);
                --stroke: rgba(16, 42, 67, 0.18);
                --accent: #0f766e;
                --accent-2: #f59e0b;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 16%, rgba(15, 118, 110, 0.16), transparent 38%),
                    radial-gradient(circle at 92% 14%, rgba(245, 158, 11, 0.2), transparent 34%),
                    linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 72%);
                color: var(--ink);
                font-family: 'DM Sans', sans-serif;
            }

            .main .block-container {
                max-width: 1080px;
                padding-top: 1.1rem;
                padding-bottom: 2rem;
                animation: riseIn 420ms ease-out;
            }

            @keyframes riseIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .hero {
                border: 1px solid var(--stroke);
                border-radius: 20px;
                padding: 1.35rem 1.5rem;
                margin-bottom: 1rem;
                background: linear-gradient(130deg, rgba(255,255,255,0.9), rgba(255,255,255,0.68));
                box-shadow: 0 14px 40px rgba(16, 42, 67, 0.09);
            }

            .hero-kicker {
                display: inline-block;
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--accent);
                background: rgba(15, 118, 110, 0.12);
                border-radius: 999px;
                padding: 0.22rem 0.62rem;
                margin-bottom: 0.7rem;
                font-weight: 700;
            }

            .hero h1 {
                margin: 0 0 0.4rem 0;
                font-family: 'Sora', sans-serif;
                font-weight: 800;
                font-size: clamp(1.7rem, 4.1vw, 2.45rem);
                line-height: 1.16;
                color: var(--ink);
            }

            .hero p {
                margin: 0;
                color: var(--muted);
                font-size: 1.01rem;
            }

            .quick-stats {
                margin-top: 0.85rem;
                display: flex;
                gap: 0.55rem;
                flex-wrap: wrap;
            }

            .pill {
                border: 1px solid var(--stroke);
                border-radius: 999px;
                padding: 0.35rem 0.72rem;
                font-size: 0.84rem;
                color: var(--ink);
                background: rgba(255, 255, 255, 0.75);
            }

            .nav-card {
                border: 1px solid var(--stroke);
                border-radius: 18px;
                padding: 1rem;
                background: var(--card);
                box-shadow: 0 10px 26px rgba(16, 42, 67, 0.08);
                min-height: 150px;
                margin-top: 0.2rem;
            }

            .nav-card h3 {
                margin: 0 0 0.35rem 0;
                font-family: 'Sora', sans-serif;
                font-weight: 700;
                color: var(--ink);
            }

            .nav-card p {
                margin: 0;
                color: var(--muted);
                font-size: 0.95rem;
                line-height: 1.45;
            }

            .stPageLink a {
                border-radius: 12px;
                border: 1px solid rgba(15, 118, 110, 0.35);
                background: linear-gradient(135deg, rgba(15, 118, 110, 0.95), rgba(6, 95, 70, 0.95));
                color: #ffffff !important;
                font-weight: 700;
                padding: 0.5rem 0.85rem;
            }

            @media (max-width: 768px) {
                .hero { padding: 1rem; }
                .hero p { font-size: 0.95rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <section class="hero">
            <span class="hero-kicker">FaceAuth Platform</span>
            <h1>AI Face Login System</h1>
            <p>Register users, train your model, and authenticate instantly from live camera input.</p>
            <div class="quick-stats">
                <span class="pill">Registered users: {user_count}</span>
                <span class="pill">Model: KNN + Threshold Gate</span>
                <span class="pill">UI: Streamlit</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <section class="nav-card">
                <h3>Login</h3>
                <p>Authenticate an existing user through live face recognition and confidence checks.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/login.py", label="Open Login")

    with col2:
        st.markdown(
            """
            <section class="nav-card">
                <h3>Register</h3>
                <p>Capture high-quality face images, review image quality, and retrain the model from the dashboard.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/register.py", label="Open Register")


if __name__ == "__main__":
    main()
