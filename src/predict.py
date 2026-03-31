import joblib
import numpy as np

from config import MODEL_PATH, LABEL_ENCODER_PATH, CONFIDENCE_THRESHOLD
from src.preprocessing import preprocess_image, preprocess_frame


class FacePredictor:
    def __init__(self, model_path=MODEL_PATH, label_encoder_path=LABEL_ENCODER_PATH):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def predict_from_image_path(self, image_path):
        """
        Predict user from an image file path.
        Returns:
            {
                "success": bool,
                "predicted_user": str or None,
                "confidence": float,
                "accepted": bool,
                "message": str
            }
        """
        feature = preprocess_image(image_path)

        if feature is None:
            return {
                "success": False,
                "predicted_user": None,
                "confidence": 0.0,
                "accepted": False,
                "message": "No valid face detected in image."
            }

        return self._predict(feature)

    def predict_from_frame(self, frame):
        """
        Predict user from an OpenCV webcam frame.
        """
        feature = preprocess_frame(frame)

        if feature is None:
            return {
                "success": False,
                "predicted_user": None,
                "confidence": 0.0,
                "accepted": False,
                "message": "No valid face detected in frame."
            }

        return self._predict(feature)

    def _predict(self, feature):
        """
        Internal prediction logic using trained KNN model.
        """
        feature = np.array(feature).reshape(1, -1)

        predicted_class = self.model.predict(feature)[0]
        predicted_user = self.label_encoder.inverse_transform([predicted_class])[0]

        confidence = self._calculate_confidence(feature, predicted_class)
        accepted = confidence >= CONFIDENCE_THRESHOLD

        if not accepted:
            predicted_user = "Unknown"

        return {
            "success": True,
            "predicted_user": predicted_user,
            "confidence": round(confidence, 4),
            "accepted": accepted,
            "message": "Prediction successful." if accepted else "Face not confidently recognized."
        }

    def _calculate_confidence(self, feature, predicted_class):
        """
        Estimate confidence for KNN based on neighbor agreement.
        Since KNN doesn't naturally output a true probability in a reliable way for this task,
        we use predict_proba if available, otherwise neighbor voting.
        """
        try:
            probabilities = self.model.predict_proba(feature)[0]
            confidence = float(np.max(probabilities))
            return confidence
        except Exception:
            # fallback method
            neighbors = self.model.kneighbors(feature, return_distance=False)[0]
            neighbor_labels = self.model._y[neighbors]
            agreement = np.mean(neighbor_labels == predicted_class)
            return float(agreement)


if __name__ == "__main__":
    predictor = FacePredictor()

    test_image = "data/raw/test/test.jpg"  # replace with your real test image
    result = predictor.predict_from_image_path(test_image)

    print("Prediction Result:")
    print(result)
