import os
import sys
from pathlib import Path


# Для обычного запуска: runtime и ресурсы — директория проекта.
# Для .exe: runtime (запись БД) — рядом с EXE, ресурсы (шаблоны/модель) — из _MEIPASS.
if getattr(sys, "frozen", False):
    RUNTIME_DIR = Path(sys.executable).resolve().parent
    RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", RUNTIME_DIR))
else:
    RUNTIME_DIR = Path(__file__).resolve().parent
    RESOURCE_DIR = RUNTIME_DIR

BASE_DIR = RUNTIME_DIR


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "agropredict-student-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{RUNTIME_DIR / 'instance' / 'app.db'}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DATA_DIR = RESOURCE_DIR / "data"
    ML_DIR = RESOURCE_DIR / "ml"
    MODEL_PATH = ML_DIR / "model.pkl"
    METRICS_PATH = ML_DIR / "metrics.json"
    APP_DIR = RESOURCE_DIR / "app"
    APP_STATIC_DIR = APP_DIR / "static"
    APP_TEMPLATES_DIR = APP_DIR / "templates"
    INSTANCE_DIR = RUNTIME_DIR / "instance"
