from preprocessing import preprocess_image
import os
import numpy as np

def all_images(images_directory="src/data/raw"):
    X, y = [], []

    for username in os.listdir(images_directory):
        user_folder = os.path.join(images_directory, username)

        if not os.path.isdir(user_folder):
            continue

        for userimage in os.listdir(user_folder):
            imagepath = os.path.join(user_folder, userimage)

            feature = preprocess_image(imagepath)

            if feature is None:
                print("The image is not processed correctly")
                continue

            X.append(feature)
            y.append(username)

    X = np.array(X)
    y = np.array(y)

    return X, y