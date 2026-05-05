from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "crop_yield.csv"
ML_DIR = ROOT_DIR / "ml"
MODEL_PATH = ML_DIR / "model.pkl"
METRICS_PATH = ML_DIR / "metrics.json"
MODEL_COMPARISON_PATH = ML_DIR / "model_comparison.png"
FEATURE_IMPORTANCE_PATH = ML_DIR / "feature_importance.png"

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


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV файл не найден: {DATA_PATH}")

    try:
        data = pd.read_csv(DATA_PATH)
    except Exception as exc:
        raise RuntimeError(f"Ошибка чтения CSV файла: {exc}") from exc

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"В датасете отсутствуют обязательные столбцы: {missing}")

    return data


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_model_pipeline(regressor) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", regressor),
        ]
    )


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_compare_models(X_train, X_test, y_train, y_test) -> tuple[dict[str, dict[str, float]], str]:
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=220,
            random_state=42,
            max_depth=8,
            min_samples_leaf=2,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=42,
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
        ),
    }

    metrics = {}
    for model_name, regressor in models.items():
        pipeline = build_model_pipeline(regressor)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics[model_name] = evaluate_predictions(y_test, predictions)

    best_model_name = sorted(
        metrics,
        key=lambda name: (-metrics[name]["R2"], metrics[name]["RMSE"]),
    )[0]
    return metrics, best_model_name


def create_final_model(best_model_name: str, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=220,
            random_state=42,
            max_depth=8,
            min_samples_leaf=2,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=42,
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
        ),
    }
    final_pipeline = build_model_pipeline(regressors[best_model_name])
    final_pipeline.fit(X, y)
    return final_pipeline


def save_metrics(metrics: dict[str, dict[str, float]], best_model_name: str, row_count: int) -> None:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": row_count,
        "target": TARGET_COLUMN,
        "best_model": best_model_name,
        "selection_rule": "maximum R2, then minimum RMSE",
        "models": metrics,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=4, ensure_ascii=False), encoding="utf-8")


def plot_model_comparison(metrics: dict[str, dict[str, float]]) -> None:
    model_names = list(metrics.keys())
    r2_values = [metrics[name]["R2"] for name in model_names]
    rmse_values = [metrics[name]["RMSE"] for name in model_names]
    x = np.arange(len(model_names))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(11, 6))
    bars_r2 = ax1.bar(x - width / 2, r2_values, width, label="R²", color="#5f8d4e")
    ax1.set_ylabel("R² score")
    ax1.set_ylim(min(0, min(r2_values) - 0.1), 1.05)

    ax2 = ax1.twinx()
    bars_rmse = ax2.bar(x + width / 2, rmse_values, width, label="RMSE", color="#d89b35")
    ax2.set_ylabel("RMSE")

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=10, ha="right")
    ax1.set_title("Сравнение моделей прогнозирования урожайности")
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend([bars_r2, bars_rmse], ["R²", "RMSE"], loc="upper left")
    fig.tight_layout()
    fig.savefig(MODEL_COMPARISON_PATH, dpi=180)
    plt.close(fig)


def plot_feature_importance(pipeline: Pipeline) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-15:]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(feature_names[top_indices], importances[top_indices], color="#5f8d4e")
        ax.set_title("Важность признаков лучшей модели")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=180)
        plt.close(fig)
        return

    # LinearRegression has coefficients, but not feature_importances_. For a fair dashboard,
    # create a clear explanatory chart instead of failing the training script.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "Лучшая модель не поддерживает feature_importances_",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="#315c2b",
    )
    ax.text(
        0.5,
        0.38,
        "Для LinearRegression можно анализировать коэффициенты,\nно они не являются прямым аналогом важности признаков.",
        ha="center",
        va="center",
        fontsize=12,
        color="#6c765f",
    )
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=42,
    )

    metrics, best_model_name = train_and_compare_models(X_train, X_test, y_train, y_test)
    final_model = create_final_model(best_model_name, X, y)

    joblib.dump(final_model, MODEL_PATH)
    save_metrics(metrics, best_model_name, row_count=len(data))
    plot_model_comparison(metrics)
    plot_feature_importance(final_model)

    print("Обучение завершено")
    print(f"Лучшая модель: {best_model_name}")
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Метрики сохранены: {METRICS_PATH}")


if __name__ == "__main__":
    main()
