import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from config import LABEL_ENCODER_PATH, MODEL_PATH
    from src.feature_engineering import all_images
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from config import LABEL_ENCODER_PATH, MODEL_PATH
    from src.feature_engineering import all_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_model(test_size: float = 0.2, random_state: int = 42) -> None:
    """Evaluate the saved face recognition model on the current dataset."""
    X, y = all_images()

    if len(X) == 0:
        print("No data available for evaluation.")
        return

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if len(np.unique(y_encoded)) < 2:
        print("Need at least two users to evaluate the model.")
        return

    try:
        _, X_test, _, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )
    except ValueError:
        _, X_test, _, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
        )

    model = joblib.load(MODEL_PATH)

    if LABEL_ENCODER_PATH.exists():
        saved_label_encoder = joblib.load(LABEL_ENCODER_PATH)
        class_names = list(saved_label_encoder.classes_)
    else:
        class_names = list(label_encoder.classes_)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))


if __name__ == "__main__":
    evaluate_model()
