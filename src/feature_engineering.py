import numpy as np

from config import RAW_DIR
from src.preprocessing import preprocess_image


def all_images(images_directory=RAW_DIR):
    """Load all user images and convert them into ML-ready features and labels."""
    X, y = [], []

    if not images_directory.exists():
        print(f"Images directory not found: {images_directory}")
        return np.array(X), np.array(y)

    for user_folder in images_directory.iterdir():
        if not user_folder.is_dir():
            continue

        username = user_folder.name

        for image_path in user_folder.iterdir():
            if not image_path.is_file():
                continue

            feature = preprocess_image(image_path)

            if feature is None:
                print(f"Skipping invalid image: {image_path}")
                continue

            X.append(feature)
            y.append(username)

    return np.array(X), np.array(y)
