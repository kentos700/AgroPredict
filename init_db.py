from pathlib import Path

from app import create_app, db
from config import BASE_DIR


app = create_app()


if __name__ == "__main__":
    Path(BASE_DIR / "instance").mkdir(parents=True, exist_ok=True)
    with app.app_context():
        db.create_all()
    print("База данных создана: instance/app.db")
