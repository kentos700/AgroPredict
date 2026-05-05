from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "ml" / "model.pkl"

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


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Файл ml/model.pkl не найден. Запустите python ml/train_model.py")

    sample = {
        "Region": "Samarkand",
        "Crop": "Wheat",
        "Rainfall": 430,
        "Temperature": 23.5,
        "Humidity": 58,
        "Soil_Type": "Loam",
        "Soil_pH": 6.8,
        "Nitrogen": 82,
        "Phosphorus": 43,
        "Potassium": 128,
        "Fertilizer_Used": "Yes",
        "Irrigation": "Yes",
        "Area": 15,
        "Season": "Spring",
    }

    model = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(frame)[0])
    print(f"Пример прогноза урожайности: {prediction:.2f} т/га")


if __name__ == "__main__":
    main()
