import joblib
import numpy as np

from config import MODEL_PATH, LABEL_ENCODER_PATH, KNN_DISTANCE_THRESHOLD
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
        predicted_user = self.label_encoder.inverse_transform([predicted_class])[
            0]

        confidence, average_distance = self._calculate_confidence(
            feature, predicted_class)
        accepted = average_distance <= KNN_DISTANCE_THRESHOLD

        if not accepted:
            predicted_user = "Unknown"

        return {
            "success": True,
            "predicted_user": predicted_user,
            "confidence": round(confidence, 4),
            "distance": round(average_distance, 4),
            "accepted": accepted,
            "message": "Prediction successful." if accepted else "Face not confidently recognized."
        }

    def _calculate_confidence(self, feature, predicted_class):
        """
        Estimate confidence from the average distance to the three nearest neighbors.
        This is stricter than raw vote confidence because a label only counts as accepted
        when the face is actually close to known examples.
        """
        distances, neighbors = self.model.kneighbors(feature, n_neighbors=3)
        average_distance = float(np.mean(distances[0]))

        if KNN_DISTANCE_THRESHOLD <= 0:
            confidence = 0.0
        else:
            confidence = float(
                np.exp(-average_distance / KNN_DISTANCE_THRESHOLD))

        return confidence, average_distance


if __name__ == "__main__":
    predictor = FacePredictor()

    test_image = "data/raw/test/test.jpg"  # replace with your real test image
    result = predictor.predict_from_image_path(test_image)

    print("Prediction Result:")
    print(result)
