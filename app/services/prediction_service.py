from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import url_for

from app import db
from app.models import PredictionHistory
from app.services.ml_service import predict_yield
from app.utils import display_prediction_parameters, form_payload_to_model_input, translate_value
from config import Config


def create_prediction(payload: dict[str, Any]) -> tuple[PredictionHistory, dict[str, Any]]:
    """Run model prediction and persist the result in SQLite."""
    model_input = form_payload_to_model_input(payload)
    predicted_value = round(predict_yield(model_input), 2)

    record = PredictionHistory(
        region=model_input["Region"],
        crop=model_input["Crop"],
        rainfall=float(model_input["Rainfall"]),
        temperature=float(model_input["Temperature"]),
        humidity=float(model_input["Humidity"]),
        soil_type=model_input["Soil_Type"],
        soil_ph=float(model_input["Soil_pH"]),
        nitrogen=float(model_input["Nitrogen"]),
        phosphorus=float(model_input["Phosphorus"]),
        potassium=float(model_input["Potassium"]),
        fertilizer_used=model_input["Fertilizer_Used"],
        irrigation=model_input["Irrigation"],
        area=float(model_input["Area"]),
        season=model_input["Season"],
        predicted_yield=predicted_value,
    )

    try:
        db.session.add(record)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise RuntimeError("Не удалось сохранить прогноз в базе данных.")

    return record, model_input


def get_prediction_history() -> list[PredictionHistory]:
    return PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()


def make_result_payload(record: PredictionHistory, model_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_yield": record.predicted_yield,
        "crop": translate_value(record.crop),
        "region": translate_value(record.region),
        "season": translate_value(record.season),
        "parameters": display_prediction_parameters(model_input),
    }


def load_metrics() -> dict[str, Any] | None:
    metrics_path = Path(Config.METRICS_PATH)
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def get_dashboard_context() -> dict[str, Any]:
    metrics = load_metrics()
    model_comparison_path = Path(Config.ML_DIR) / "model_comparison.png"
    feature_importance_path = Path(Config.ML_DIR) / "feature_importance.png"

    return {
        "metrics": metrics,
        "model_comparison_url": url_for("main.ml_artifact", filename="model_comparison.png")
        if model_comparison_path.exists()
        else None,
        "feature_importance_url": url_for("main.ml_artifact", filename="feature_importance.png")
        if feature_importance_path.exists()
        else None,
    }
