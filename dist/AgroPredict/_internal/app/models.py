from datetime import datetime, timezone

from app import db


class PredictionHistory(db.Model):
    """Stores every yield prediction made by the web application."""

    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(80), nullable=False)
    crop = db.Column(db.String(80), nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    soil_type = db.Column(db.String(80), nullable=False)
    soil_ph = db.Column(db.Float, nullable=False)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    fertilizer_used = db.Column(db.String(20), nullable=False)
    irrigation = db.Column(db.String(20), nullable=False)
    area = db.Column(db.Float, nullable=False)
    season = db.Column(db.String(40), nullable=False)
    predicted_yield = db.Column(db.Float, nullable=False)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def __repr__(self):
        return f"<PredictionHistory {self.crop} {self.region} {self.predicted_yield:.2f}>"
