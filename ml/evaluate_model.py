from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "crop_yield.csv"
MODEL_PATH = ROOT_DIR / "ml" / "model.pkl"
METRICS_PATH = ROOT_DIR / "ml" / "metrics.json"
TARGET_COLUMN = "Yield"
FEATURE_COLUMNS = [
    "Region",
    "Crop",
    "Rainfall",
    "Temperature",
    "Humidity",
    "Soil_Type",
    "Soil_pH",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Fertilizer_Used",
    "Irrigation",
    "Area",
    "Season",
]


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Файл ml/model.pkl не найден. Запустите python ml/train_model.py")

    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)
    holdout_metrics = calculate_metrics(y_test, predictions)

    print("Оценка сохраненной модели на контрольной выборке:")
    for name, value in holdout_metrics.items():
        print(f"{name}: {value:.4f}")

    if METRICS_PATH.exists():
        saved_report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        print("\nМетрики, сохраненные при обучении:")
        print(json.dumps(saved_report, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
