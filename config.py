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
CAPTURE_COUNT = 15
CONFIDENCE_THRESHOLD = 0.75
TO_GRAY = True


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
