
import joblib
from pathlib import Path

from .preprocessing import preprocess


MODEL_DIR = Path(__file__).resolve().parents[2] / "Models" / "Text"


def load_models():
    models = {
        "logistic_regression": joblib.load(
            MODEL_DIR / "logistic_regression.pkl"
        ),
        "linear_svm": joblib.load(
            MODEL_DIR / "linear_svm.pkl"
        ),
        "voting_classifier": joblib.load(
            MODEL_DIR / "voting_classifier.pkl"
        ),
        "vectorizer": joblib.load(
            MODEL_DIR / "tfidf_vectorizer.joblib"
        )
    }
    return models


def predict(text, model_name, models):
    """
    Predict sentiment.

    Args:
        text: User review.
        model_name: Selected sentiment classification model.
        models: Dictionary containing loaded models.

    Returns:
        Dictionary containing model name and prediction.
    """

    # TODO: Call preprocess()
    # TODO: Select the requested model
    # TODO: Run model.predict()
    # TODO: Map prediction to Positive or Negative
    # TODO: Return prediction dictionary
    pass
