from __future__ import annotations

from typing import Any

import pandas as pd


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

NUMERIC_COLUMNS = [
    "Rainfall",
    "Temperature",
    "Humidity",
    "Soil_pH",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Area",
]

CATEGORICAL_COLUMNS = [
    "Region",
    "Crop",
    "Soil_Type",
    "Fertilizer_Used",
    "Irrigation",
    "Season",
]

FORM_TO_MODEL_COLUMNS = {
    "region": "Region",
    "crop": "Crop",
    "rainfall": "Rainfall",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "soil_type": "Soil_Type",
    "soil_ph": "Soil_pH",
    "nitrogen": "Nitrogen",
    "phosphorus": "Phosphorus",
    "potassium": "Potassium",
    "fertilizer_used": "Fertilizer_Used",
    "irrigation": "Irrigation",
    "area": "Area",
    "season": "Season",
}

DISPLAY_LABELS = {
    "Region": "Регион",
    "Crop": "Категория культуры",
    "Rainfall": "Осадки, мм",
    "Temperature": "Температура, °C",
    "Humidity": "Влажность, %",
    "Soil_Type": "Тип почвы",
    "Soil_pH": "pH почвы",
    "Nitrogen": "Азот, мг/кг",
    "Phosphorus": "Фосфор, мг/кг",
    "Potassium": "Калий, мг/кг",
    "Fertilizer_Used": "Удобрения",
    "Irrigation": "Орошение",
    "Area": "Площадь, га",
    "Season": "Сезон",
}

VALUE_TRANSLATIONS = {
    "Yes": "Да",
    "No": "Нет",
    "Spring": "Весна",
    "Summer": "Лето",
    "Autumn": "Осень",
    "Winter": "Зима",
    "Loam": "Суглинистая",
    "Clay": "Глинистая",
    "Sandy": "Песчаная",
    "Silt": "Илистая",
    "Black": "Чернозём",
    "Saline": "Засоленная",
    "Wheat": "Пшеница",
    "Cotton": "Хлопок",
    "Rice": "Рис",
    "Corn": "Кукуруза",
    "Potato": "Картофель",
    "Barley": "Ячмень",
    "Tomato": "Томат",
    "Tashkent": "Ташкент",
    "Samarkand": "Самарканд",
    "Fergana": "Фергана",
    "Andijan": "Андижан",
    "Bukhara": "Бухара",
    "Khorezm": "Хорезм",
    "Kashkadarya": "Кашкадарья",
    "Jizzakh": "Джизак",
    "Namangan": "Наманган",
    "Surkhandarya": "Сурхандарья",
    "Karakalpakstan": "Республика Каракалпакстан",
}


def translate_value(value: Any) -> str:
    """Переводит известные категориальные значения на русский для UI."""
    return VALUE_TRANSLATIONS.get(str(value), str(value))


def build_prediction_dataframe(model_input: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row DataFrame in the exact order expected by the ML pipeline."""
    row = {column: model_input[column] for column in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def form_payload_to_model_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert validated form field names into dataset/model column names."""
    model_input = {}
    for form_name, model_name in FORM_TO_MODEL_COLUMNS.items():
        if form_name not in payload:
            raise ValueError(f"Отсутствует поле формы: {form_name}")
        model_input[model_name] = payload[form_name]
    return model_input


def display_prediction_parameters(model_input: dict[str, Any]) -> list[tuple[str, str]]:
    """Return translated labels and values for result pages."""
    rows = []
    for column in FEATURE_COLUMNS:
        value = model_input.get(column)
        if isinstance(value, float):
            value_text = f"{value:.2f}".rstrip("0").rstrip(".")
        else:
            value_text = str(value)
        rows.append((DISPLAY_LABELS[column], translate_value(value_text)))
    return rows
