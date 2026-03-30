import os

import cv2

from config import CAPTURE_COUNT, user_raw_dir


def capture_images(username: str) -> None:
    """Capture face images into the canonical raw data folder."""

    save_directory = user_raw_dir(username)
    os.makedirs(save_directory, exist_ok=True)

    capture = cv2.VideoCapture(0)
    count = 0

    while True:
        ref, imageframe = capture.read()

        if not ref:
            print("the image is not working")
            break

        cv2.imshow("Webcame", imageframe)
        keytype = cv2.waitKey(1)

        if keytype == 27:
            break

        if keytype == 32:
            if count < CAPTURE_COUNT:
                filename = os.path.join(
                    save_directory, f"image_of_{username}_{count + 1}.jpg")
                cv2.imwrite(filename, imageframe)
                count += 1
            else:
                break

    capture.release()
