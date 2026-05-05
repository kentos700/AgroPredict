import threading
import webbrowser
from pathlib import Path

from app import create_app, db
from config import BASE_DIR


def open_browser() -> None:
    # Открываем браузер после старта локального сервера.
    webbrowser.open("http://127.0.0.1:5000")


def ensure_database() -> None:
    # Гарантируем наличие таблиц даже при первом запуске из EXE.
    Path(BASE_DIR / "instance").mkdir(parents=True, exist_ok=True)
    app = create_app()
    with app.app_context():
        db.create_all()


if __name__ == "__main__":
    ensure_database()
    app = create_app()
    threading.Timer(1.0, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
