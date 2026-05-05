from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NumericRule:
    label: str
    min_value: float | None = None
    max_value: float | None = None
    step: str = "0.01"
    placeholder: str = ""
    help_text: str = ""


class PredictionForm:
    """Лёгкая форма для Flask с ручной валидацией без внешних библиотек."""

    select_fields = {
        "region": {
            "label": "Регион",
            "choices": [
                "Tashkent",
                "Samarkand",
                "Fergana",
                "Andijan",
                "Bukhara",
                "Khorezm",
                "Kashkadarya",
                "Jizzakh",
                "Namangan",
                "Surkhandarya",
                "Karakalpakstan",
            ],
        },
        "crop": {
            "label": "Категория культуры",
            "choices": ["Wheat", "Cotton", "Rice", "Corn", "Potato", "Barley", "Tomato"],
        },
        "soil_type": {
            "label": "Тип почвы",
            "choices": ["Loam", "Clay", "Sandy", "Silt", "Black", "Saline"],
        },
        "fertilizer_used": {
            "label": "Использовались удобрения",
            "choices": ["Yes", "No"],
        },
        "irrigation": {
            "label": "Орошение",
            "choices": ["Yes", "No"],
        },
        "season": {
            "label": "Сезон",
            "choices": ["Spring", "Summer", "Autumn", "Winter"],
        },
    }

    choice_labels = {
        "region": {
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
        },
        "crop": {
            "Wheat": "Пшеница",
            "Cotton": "Хлопок",
            "Rice": "Рис",
            "Corn": "Кукуруза",
            "Potato": "Картофель",
            "Barley": "Ячмень",
            "Tomato": "Томат",
        },
        "soil_type": {
            "Loam": "Суглинистая",
            "Clay": "Глинистая",
            "Sandy": "Песчаная",
            "Silt": "Илистая",
            "Black": "Чернозём",
            "Saline": "Засоленная",
        },
        "season": {
            "Spring": "Весна",
            "Summer": "Лето",
            "Autumn": "Осень",
            "Winter": "Зима",
        },
        "fertilizer_used": {
            "Yes": "Да",
            "No": "Нет",
        },
        "irrigation": {
            "Yes": "Да",
            "No": "Нет",
        },
    }

    numeric_fields = {
        "rainfall": NumericRule("Осадки, мм", 0, 2500, placeholder="Например: 420"),
        "temperature": NumericRule("Средняя температура, °C", -20, 60, placeholder="Например: 24.5"),
        "humidity": NumericRule("Влажность, %", 0, 100, placeholder="Например: 62"),
        "soil_ph": NumericRule("pH почвы", 3.5, 9.5, placeholder="Например: 6.8"),
        "nitrogen": NumericRule("Азот, мг/кг", 0, 300, placeholder="Например: 78"),
        "phosphorus": NumericRule("Фосфор, мг/кг", 0, 250, placeholder="Например: 44"),
        "potassium": NumericRule("Калий, мг/кг", 0, 350, placeholder="Например: 120"),
        "area": NumericRule("Площадь, га", 0.01, 100000, placeholder="Например: 12.5"),
    }

    field_order = [
        "region",
        "crop",
        "rainfall",
        "temperature",
        "humidity",
        "soil_type",
        "soil_ph",
        "nitrogen",
        "phosphorus",
        "potassium",
        "fertilizer_used",
        "irrigation",
        "area",
        "season",
    ]

    def __init__(self, form_data: Any | None = None):
        self.raw_data = {}
        self.cleaned_data = {}
        self.errors: dict[str, list[str]] = {}
        for field in self.field_order:
            self.raw_data[field] = ""
            if form_data is not None:
                self.raw_data[field] = str(form_data.get(field, "")).strip()

    def value(self, field: str) -> str:
        return self.raw_data.get(field, "")

    def label(self, field: str) -> str:
        if field in self.select_fields:
            return self.select_fields[field]["label"]
        return self.numeric_fields[field].label

    def choices(self, field: str) -> list[str]:
        return list(self.select_fields[field]["choices"])

    def choice_label(self, field: str, value: str) -> str:
        return self.choice_labels.get(field, {}).get(value, value)

    def numeric_rule(self, field: str) -> NumericRule:
        return self.numeric_fields[field]

    def validate(self) -> bool:
        self.errors = {}
        self.cleaned_data = {}

        for field in self.field_order:
            value = self.raw_data.get(field, "")
            if value == "":
                self.add_error(field, "Поле обязательно для заполнения.")
                continue

            if field in self.select_fields:
                self._validate_select(field, value)
            else:
                self._validate_number(field, value)

        return not self.errors

    def add_error(self, field: str, message: str) -> None:
        self.errors.setdefault(field, []).append(message)

    def _validate_select(self, field: str, value: str) -> None:
        choices = self.select_fields[field]["choices"]
        if value not in choices:
            self.add_error(field, "Выберите значение из списка.")
            return
        self.cleaned_data[field] = value

    def _validate_number(self, field: str, value: str) -> None:
        rule = self.numeric_fields[field]
        try:
            number = float(value.replace(",", "."))
        except ValueError:
            self.add_error(field, "Введите корректное числовое значение.")
            return

        if rule.min_value is not None and number < rule.min_value:
            self.add_error(field, f"Значение должно быть не меньше {rule.min_value}.")
            return
        if rule.max_value is not None and number > rule.max_value:
            self.add_error(field, f"Значение должно быть не больше {rule.max_value}.")
            return

        self.cleaned_data[field] = number
        self.raw_data[field] = str(number)

    def to_payload(self) -> dict[str, Any]:
        if not self.cleaned_data:
            raise ValueError("Форма не была успешно проверена.")
        return dict(self.cleaned_data)
