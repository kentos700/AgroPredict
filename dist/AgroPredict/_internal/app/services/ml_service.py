from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from app.utils import build_prediction_dataframe
from config import Config


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact is missing."""


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained scikit-learn pipeline."""
    model_path = Path(Config.MODEL_PATH)
    if not model_path.exists():
        raise ModelNotFoundError(
            "Файл модели ml/model.pkl не найден. Сначала выполните команду: python ml/train_model.py"
        )
    return joblib.load(model_path)


def predict_yield(model_input: dict) -> float:
    """Predict crop yield for a validated dictionary of model features."""
    model = load_model()
    frame = build_prediction_dataframe(model_input)
    prediction = float(model.predict(frame)[0])
    return max(prediction, 0.0)
