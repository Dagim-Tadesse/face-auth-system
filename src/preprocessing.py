import cv2

from config import IMAGE_SIZE

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def preprocess_image(image_path, target_size=IMAGE_SIZE, to_gray=True):
    """Load an image, detect the largest face, resize, normalize, and flatten."""
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
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
        print(f"No face detected in: {image_path}")
        return None

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face_region = processed_image[y:y+h, x:x+w]
    resized_face = cv2.resize(face_region, target_size)
    normalized_face = resized_face.astype("float32") / 255.0
    flattened_face = normalized_face.flatten()

    return flattened_face


def preprocess_frame(frame, target_size=IMAGE_SIZE, to_gray=True):
    """Preprocess a webcam frame for prediction."""
    if frame is None:
        return None

    if to_gray:
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        processed_frame = frame

    faces = face_detector.detectMultiScale(
        processed_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face_region = processed_frame[y:y+h, x:x+w]
    resized_face = cv2.resize(face_region, target_size)
    normalized_face = resized_face.astype("float32") / 255.0

    return normalized_face.flatten()
