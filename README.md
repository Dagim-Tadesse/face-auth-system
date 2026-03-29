# Face Auth System

This project is a face-authentication prototype with a shared folder contract:

- Raw captures live in `data/raw/<username>/`
- Processed artifacts live in `data/processed/`
- Trained models live in `models/`
- The canonical model path is `models/face_model.pkl`

## Module Interfaces

These are the expected inputs and outputs for each module so the team can work independently without hardcoding different paths or names.

### `src/data_collection.py`

- Exposes: `capture_images(username: str) -> None`
- Responsibility: capture webcam images for one user and save them into `data/raw/<username>/`
- Output: raw image files only

### `src/preprocessing.py`

- Exposes: `preprocess_image(image_path: str, target_size=(64, 64), to_gray=True)`
- Responsibility: load one image, detect a face, crop it, resize it, normalize it, and flatten it
- Output: one feature vector or `None` if the image cannot be processed

### `src/feature_engineering.py`

- Exposes: `all_images(images_directory=RAW_DIR)`
- Responsibility: walk through `data/raw/`, preprocess each image, and return training arrays
- Output: `X` feature matrix and `y` label array

### `src/train.py`

- Expected interface: accept `X` and `y`, train the model, and save it to `models/face_model.pkl`
- Output: persisted model file

### `src/predict.py`

- Expected interface: accept one face image or feature vector and return a predicted label plus confidence
- Acceptance rule: the prediction should be treated as valid only if confidence is at or above the shared threshold in `config.py`

### `src/evaluate.py`

- Expected interface: measure model performance on held-out data and report accuracy or similar metrics

### `app/main.py` and `app/pages/*`

- Expected interface: provide the Streamlit shell for registration and login flows
- Responsibility: call into the shared `src/` modules instead of re-implementing capture, preprocessing, or prediction logic

## Setup

Install the dependencies first, then run the app or pipeline modules from the project root.

```bash
pip install -r requirements.txt
```

## Notes

- Keep face images out of version control.
- Keep all path values in `config.py`.
- Do not introduce new folder names for raw captures unless the shared contract changes for everyone.
