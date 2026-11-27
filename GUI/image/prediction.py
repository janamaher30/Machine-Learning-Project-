
import joblib
from pathlib import Path

from .preprocessing import preprocess


MODEL_DIR = Path(__file__).resolve().parents[2] / "Models" / "Image"


def load_models():

    models = {
        "color_knn": joblib.load(MODEL_DIR / "color_knn_model.pkl"),
        "color_lr": joblib.load(MODEL_DIR / "color_lr_model.pkl"),
        "scaler": joblib.load(MODEL_DIR / "color_scaler.pkl"),
        "class_names": joblib.load(MODEL_DIR / "class_names.pkl")
    }

    return models



def predict(image, model_name, models):
    """
    Predict the flower class.

    Args:
        image: Uploaded PIL Image.
        model_name: Selected image classification model.
        models: Dictionary containing loaded models.

    Returns:
        Dictionary containing model name and prediction.
    """

    # TODO: Call preprocess()
    # TODO: Select the requested model
    # TODO: Run model.predict()
    # TODO: Map predicted class to class name
    # TODO: Return prediction dictionary
    pass
