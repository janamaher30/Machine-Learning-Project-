
import joblib
from pathlib import Path

from .preprocessing import preprocess


MODEL_DIR = Path(__file__).resolve().parents[2] / "Models" / "Numerical"


def load_models():
    models = {
        "linear_regression": joblib.load(MODEL_DIR / "linear_regression.pkl"),
        "knn": joblib.load(MODEL_DIR / "knn_model.pkl"),
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "feature_columns": joblib.load(MODEL_DIR / "feature_columns.pkl")
    }

    return models


def predict(input_data, model_name, models):
    # TODO: Call preprocess()
    # TODO: Select the requested model
    # TODO: Run model.predict()
    # TODO: Return prediction dictionary
    pass