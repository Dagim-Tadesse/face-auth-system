# FaceAuth Capstone Report

## 1. Problem Statement

Traditional username and password systems are easy to deploy but can be inconvenient and vulnerable to credential sharing. This project implements an AI-based facial authentication flow where users register face samples and later authenticate with a live camera feed.

## 2. Approach

The system uses a classical computer vision and machine learning pipeline.

- Data collection: user-specific face images are captured and stored in data/raw.
- Face detection and preprocessing: OpenCV Haar cascades detect faces, then the face is cropped, resized to 64x64, optionally converted to grayscale, normalized, and flattened.
- Feature engineering: flattened face vectors become input features X; user IDs become labels y.
- Model training: a K-Nearest Neighbors classifier is trained and saved to disk.
- Inference: a live frame is preprocessed the same way, then passed through the saved model.
- Decision rule: prediction is accepted only when distance-based confidence checks pass threshold constraints.

## 3. Model Choice and Rationale

KNN was selected because:

- It is simple and fast to prototype for small datasets.
- It works well with vectorized image features when classes are limited.
- Distance outputs can be used directly for confidence thresholding to reduce false positives.

## 4. Results Summary

The application demonstrates:

- Face registration for multiple users.
- Model training and persistence.
- Face login with accept/deny decisions.
- Quality checks for blurry or low-quality registration images.

Measured evaluation results (latest run):

- Valid processed samples used for training/evaluation: 94
- Evaluation split size: 19 test samples
- Accuracy: 0.9474 (94.74%)

Confusion Matrix (rows = true class, columns = predicted class):

```text
[[4 0 0 0 0]
 [0 4 0 0 0]
 [0 0 3 0 1]
 [0 0 0 4 0]
 [0 0 0 0 3]]
```

Class order:

1. Ariana Grande
2. Dagim Tadesse
3. Keanu Reeves
4. Kevin Hart
5. Tom cruise

False reject / false accept behavior observed:

- False rejects were more frequent when the same person was tested with images that had different lighting/pose from registration images.
- A confusion case appeared between Keanu Reeves and Tom cruise (1 sample), which is a model-level false accept risk before threshold gating.
- With the distance threshold gate enabled in login, uncertain matches were more often denied, reducing false accepts but increasing false reject likelihood under difficult conditions.

## 5. Key Issues Encountered

- Intermittent import errors occurred when page files imported src modules before project root path bootstrap.
- Registration quality needed gating to avoid blurry frames.
- Dataset diversity had a stronger effect on performance than sample count alone.

## 6. Improvements Applied

- Shared config contract for paths and thresholds.
- Registration quality assessment for blur/brightness/contrast/face-size checks.
- Evaluation script cleanup for consistent accuracy and confusion-matrix reporting.
- Baseline automated tests for core pipeline behavior.

## 7. Limitations

- Haar-cascade plus flattened grayscale features are sensitive to lighting and pose changes.
- KNN scales poorly with large user sets.
- Current anti-spoofing is limited.

## 8. Future Improvements

- Replace handcrafted features with embedding-based face recognition.
- Add liveness detection.
- Add class-balanced data collection guidance in UI.
- Build calibration tooling for threshold tuning using validation sets.

## 9. Data Availability and Reproducibility

- The reported metrics were computed on a local dataset in `data/raw`, which is intentionally excluded from the public repository due privacy/copyright and repository-size constraints.
- Evaluation was executed on April 6, 2026 using this repository's preprocessing, training, and evaluation pipeline.
- The measured results in this report are from the current code path with 94 valid processed samples; some raw images were skipped when face detection failed.
- Because raw images are private, full end-to-end reproduction from GitHub alone is not possible.
- Reproducible artifacts that are available in the repository include: code, model pipeline logic, preprocessing rules, configuration values, and evaluation script behavior.
- For assessor verification, the exact command flow used was: `python src/train.py` followed by `python src/evaluate.py` in the project virtual environment.

## 10. Conclusion

The project meets the objective of an end-to-end ML authentication product with registration, training, evaluation, and live login in Streamlit. Performance is functional for a capstone prototype and can be significantly improved with richer user data and embedding-based models.
