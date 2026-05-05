from pathlib import Path

from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

from config import Config
from app.utils import translate_value


db = SQLAlchemy()


def create_app(config_class=Config):
    """Create and configure the Flask application."""
    static_dir = Path(config_class.APP_STATIC_DIR)
    templates_dir = Path(config_class.APP_TEMPLATES_DIR)
    instance_dir = Path(config_class.INSTANCE_DIR)

    app = Flask(
        __name__,
        instance_relative_config=True,
        instance_path=str(instance_dir),
        static_folder=str(static_dir),
        template_folder=str(templates_dir),
    )
    app.config.from_object(config_class)

    instance_dir.mkdir(parents=True, exist_ok=True)
    ml_dir = Path(app.config["ML_DIR"])
    if not ml_dir.exists():
        ml_dir.mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    from app.routes import main_bp

    app.register_blueprint(main_bp)
    app.jinja_env.filters["translate_value"] = translate_value

    @app.errorhandler(404)
    def not_found_error(error):
        return (
            render_template(
                "error.html",
                title="Страница не найдена",
                message="Запрошенная страница не существует или была перемещена.",
                status_code=404,
            ),
            404,
        )

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return (
            render_template(
                "error.html",
                title="Ошибка сервера",
                message="Во время обработки запроса произошла внутренняя ошибка.",
                status_code=500,
            ),
            500,
        )

    return app
