import cv2

from config import (
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_DETECTION_MIN_SIZE,
    FACE_DETECTION_SCALE_FACTOR,
    FACE_EYE_DETECTION_MIN_NEIGHBORS,
    IMAGE_SIZE,
    MAX_FACE_BRIGHTNESS,
    MIN_FACE_AREA_RATIO,
    MIN_FACE_BRIGHTNESS,
    MIN_FACE_CONTRAST,
    MIN_FACE_SHARPNESS,
)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def _detect_valid_faces(image, to_gray=True):
    if image is None:
        return []

    processed_image = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY) if to_gray else image
    faces = face_detector.detectMultiScale(
        processed_image,
        scaleFactor=FACE_DETECTION_SCALE_FACTOR,
        minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
        minSize=FACE_DETECTION_MIN_SIZE,
    )

    valid_faces = []
    for x, y, w, h in faces:
        face_region = processed_image[y:y + h, x:x + w]
        eyes = eye_detector.detectMultiScale(
            face_region,
            scaleFactor=1.1,
            minNeighbors=FACE_EYE_DETECTION_MIN_NEIGHBORS,
            minSize=(15, 15),
        )

        if len(eyes) > 0:
            valid_faces.append((x, y, w, h))

    return valid_faces


def _detect_and_crop_face(image, target_size=IMAGE_SIZE, to_gray=True):
    """Extract and resize the largest detected face from an image.

    Returns the cropped face region as uint8, or None if no face found.
    This is a private helper function used by the public preprocessing functions.
    """
    if image is None:
        return None

    if to_gray:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        processed_image = image

    faces = _detect_valid_faces(image, to_gray=to_gray)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_region = processed_image[y:y+h, x:x+w]
    resized_face = cv2.resize(face_region, target_size)

    return resized_face


def preprocess_image(image_path, target_size=IMAGE_SIZE, to_gray=True):
    """Load an image, detect the largest face, resize, normalize, and flatten."""
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    cropped_face = _detect_and_crop_face(image, target_size, to_gray)
    if cropped_face is None:
        print(f"No face detected in: {image_path}")
        return None

    normalized_face = cropped_face.astype("float32") / 255.0
    return normalized_face.flatten()


def preprocess_frame(frame, target_size=IMAGE_SIZE, to_gray=True):
    """Preprocess a webcam frame for prediction."""
    cropped_face = _detect_and_crop_face(frame, target_size, to_gray)
    if cropped_face is None:
        return None

    normalized_face = cropped_face.astype("float32") / 255.0
    return normalized_face.flatten()


def get_preprocessed_face_image(frame, target_size=IMAGE_SIZE, to_gray=True):
    """Preprocess a frame and return the processed face image (2D array, not flattened).

    This is useful for saving preprocessed face images during registration.
    Returns the grayscale, cropped, resized face as a uint8 image ready for saving.
    """
    return _detect_and_crop_face(frame, target_size, to_gray)


def assess_face_quality(image, face_bbox=None):
    """Return quality verdict and details for one detected face.

    The function checks face size in frame, blur (Laplacian variance),
    brightness, and contrast. It returns (is_good, reasons, metrics).
    """
    if image is None:
        return False, ["Empty frame."], {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = gray.shape[:2]
    frame_area = float(frame_h * frame_w)

    face = face_bbox
    if face is None:
        faces = _detect_valid_faces(image, to_gray=True)
        if len(faces) == 0:
            return False, ["No clear frontal face detected."], {}
        face = max(faces, key=lambda f: f[2] * f[3])

    x, y, w, h = face
    face_gray = gray[y:y + h, x:x + w]
    if face_gray.size == 0:
        return False, ["Face region is invalid."], {}

    face_area_ratio = (w * h) / frame_area if frame_area else 0.0
    sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    brightness = float(face_gray.mean())
    contrast = float(face_gray.std())

    reasons = []
    if face_area_ratio < MIN_FACE_AREA_RATIO:
        reasons.append("Move closer to the camera.")
    if sharpness < MIN_FACE_SHARPNESS:
        reasons.append("Image is blurry. Hold still and refocus.")
    if brightness < MIN_FACE_BRIGHTNESS:
        reasons.append("Lighting is too dark. Increase light on your face.")
    if brightness > MAX_FACE_BRIGHTNESS:
        reasons.append("Lighting is too bright. Reduce glare or backlight.")
    if contrast < MIN_FACE_CONTRAST:
        reasons.append("Face details are too flat. Improve lighting angle.")

    metrics = {
        "face_area_ratio": face_area_ratio,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
    }

    return len(reasons) == 0, reasons, metrics
