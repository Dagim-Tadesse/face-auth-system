import os
import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LABEL_ENCODER_PATH, MODEL_PATH, MODELS_DIR
from src.feature_engineering import all_images


def train_model() -> None:
    print("Loading and preprocessing images...")
    X, y = all_images()

    if len(X) == 0:
        print("No training data found.")
        return

    print(f"Dataset loaded: {len(X)} samples")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    train_model()
