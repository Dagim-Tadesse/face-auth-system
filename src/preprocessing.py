import cv2
import numpy as np
import os

from config import IMAGE_SIZE

face_crop = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_image(image_path, target_size=IMAGE_SIZE, to_gray=True):
    image_to_extract = cv2.imread(image_path)

    if image_to_extract is None:
        print("The image is not read")
        return None

    if to_gray:
        gray = cv2.cvtColor(image_to_extract, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_to_extract

    faces = face_crop.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected")
        return None

    x, y, w, h = faces[0]

    face_image_after = gray[y:y+h, x:x+w]

    face_resize = cv2.resize(face_image_after, target_size)

    face_normalized = face_resize / 255.0

    face_flatten = face_normalized.flatten()

    return face_flatten
