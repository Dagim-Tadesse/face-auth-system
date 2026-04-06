from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def camera_image_to_frame(camera_image) -> np.ndarray | None:
    """Convert a Streamlit camera upload into an OpenCV BGR frame."""

    if camera_image is None:
        return None

    bytes_data = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
    return cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
