from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, send_from_directory, session, url_for

from app.forms import PredictionForm
from app.services.ml_service import ModelNotFoundError
from app.services.prediction_service import (
    create_prediction,
    get_dashboard_context,
    get_prediction_history,
    make_result_payload,
)
from config import Config


main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html", title="AgroPredict")


@main_bp.route("/predict", methods=["GET", "POST"])
def predict():
    form = PredictionForm(request.form if request.method == "POST" else None)

    if request.method == "POST":
        if form.validate():
            try:
                record, model_input = create_prediction(form.to_payload())
            except ModelNotFoundError as exc:
                return render_template(
                    "error.html",
                    title="Модель не найдена",
                    message=str(exc),
                    status_code=503,
                ), 503
            except ValueError as exc:
                return render_template(
                    "error.html",
                    title="Ошибка данных",
                    message=str(exc),
                    status_code=400,
                ), 400
            except RuntimeError as exc:
                return render_template(
                    "error.html",
                    title="Ошибка базы данных",
                    message=str(exc),
                    status_code=500,
                ), 500

            session["last_prediction"] = make_result_payload(record, model_input)
            flash("Прогноз успешно рассчитан и сохранен в истории.", "success")
            return redirect(url_for("main.result"))

        flash("Проверьте форму: некоторые поля заполнены некорректно.", "warning")

    return render_template("predict.html", title="Прогноз урожайности", form=form)


@main_bp.route("/result")
def result():
    result_payload = session.get("last_prediction")
    if not result_payload:
        flash("Сначала заполните форму прогнозирования.", "info")
        return redirect(url_for("main.predict"))
    return render_template("result.html", title="Результат прогноза", result=result_payload)


@main_bp.route("/history")
def history():
    records = get_prediction_history()
    return render_template("history.html", title="История прогнозов", records=records)


@main_bp.route("/dashboard")
def dashboard():
    context = get_dashboard_context()
    return render_template("dashboard.html", title="Dashboard", **context)


@main_bp.route("/error")
def error_page():
    message = request.args.get("message", "Произошла неизвестная ошибка.")
    status_code = int(request.args.get("status", 400))
    return render_template("error.html", title="Ошибка", message=message, status_code=status_code), status_code


@main_bp.route("/ml-artifacts/<path:filename>")
def ml_artifact(filename: str):
    return send_from_directory(Config.ML_DIR, filename)
