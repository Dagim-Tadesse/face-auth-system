import cv2

from config import IMAGE_SIZE

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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

    faces = face_detector.detectMultiScale(
        processed_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

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
