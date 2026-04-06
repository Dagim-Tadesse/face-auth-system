"""Shared project contract for paths and model defaults.

All modules should import these values instead of hardcoding paths or
magic numbers. The project stores raw captures under data/raw/<user>/,
intermediate artifacts under data/processed/, and the trained model at
models/face_model.pkl.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "face_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

IMAGE_SIZE = (64, 64)
CAPTURE_COUNT = 20
CONFIDENCE_THRESHOLD = 0.75
KNN_DISTANCE_THRESHOLD = 8.6
TO_GRAY = True
FACE_DETECTION_SCALE_FACTOR = 1.05
FACE_DETECTION_MIN_NEIGHBORS = 8
FACE_DETECTION_MIN_SIZE = (70, 70)
FACE_EYE_DETECTION_MIN_NEIGHBORS = 3
MIN_FACE_AREA_RATIO = 0.06
# Webcam Laplacian variance is often much lower than still-image quality;
# this value keeps a basic blur filter without blocking most live captures.
MIN_FACE_SHARPNESS = 18.0
MIN_FACE_BRIGHTNESS = 45.0
MAX_FACE_BRIGHTNESS = 210.0
MIN_FACE_CONTRAST = 16.0


def user_raw_dir(username: str) -> Path:
    """Return the canonical folder for one user's raw images."""

    return RAW_DIR / username.strip()


def ensure_project_directories() -> None:
    """Create the standard project directories if they do not exist."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def is_confident(confidence: float) -> bool:
    """Return whether a prediction meets the shared acceptance threshold."""

    return confidence >= CONFIDENCE_THRESHOLD
