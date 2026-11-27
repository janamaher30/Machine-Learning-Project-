
import joblib
from pathlib import Path

models_path = Path("Models")

model_files = [
    "Numerical/linear_regression.pkl",
    "Numerical/knn_model.pkl",
    "Numerical/scaler.pkl",
    "Numerical/feature_columns.pkl",

    "Image/color_knn_model.pkl",
    "Image/color_lr_model.pkl",
    "Image/color_scaler.pkl",
    "Image/class_names.pkl",

    "Text/logistic_regression.pkl",
    "Text/linear_svm.pkl",
    "Text/voting_classifier.pkl",
    "Text/tfidf_vectorizer.joblib"
]

for file in model_files:
    path = models_path / file

    try:
        model = joblib.load(path)
        print(f"SUCCESS: {file} -> {type(model).__name__}")

    except Exception as e:
        print(f"ERROR: {file}")
        print(e)

print("\nModel loading test finished!")
