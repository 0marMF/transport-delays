"""API de estimación de retrasos (FastAPI).

Levantar en local:  uvicorn src.api:app --reload
Documentación interactiva en  http://localhost:8000/docs

Expone el modelo como servicio: le mandas los atributos de un viaje y devuelve el retraso de
llegada estimado. El modelo se carga una sola vez.

Importante (honestidad): el modelo tiene R² ~ 0; la estimación es esencialmente el promedio
histórico y NO debe usarse para decisiones a nivel de viaje. La API demuestra el serving, no la
fiabilidad del modelo. Ver MODEL_CARD.md.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .score import DEFAULTS, load_artifact, predict

app = FastAPI(
    title="Transport Delays API",
    version="1.1.0",
    description="Estimación de retraso de llegada de transporte público. Nota: R² ~ 0, modelo no "
                "fiable a nivel de viaje (ver MODEL_CARD.md); la API demuestra el serving.",
)

_artefacto: dict | None = None


def _art() -> dict:
    # Carga perezosa: el modelo se lee del disco la primera vez y se reutiliza.
    global _artefacto
    if _artefacto is None:
        _artefacto = load_artifact()
    return _artefacto


class Trip(BaseModel):
    transport_type: str = Field(DEFAULTS["transport_type"], description="Bus, Metro, Train, Tram")
    weather_condition: str = Field(DEFAULTS["weather_condition"],
                                   description="Clear, Cloudy, Fog, Rain, Snow, Storm")
    event_type: str = Field(DEFAULTS["event_type"],
                            description="None, Concert, Festival, Parade, Protest, Sports")
    season: str = Field(DEFAULTS["season"], description="Winter, Spring, Summer, Autumn")
    temperature_C: float = DEFAULTS["temperature_C"]
    humidity_percent: float = DEFAULTS["humidity_percent"]
    wind_speed_kmh: float = DEFAULTS["wind_speed_kmh"]
    precipitation_mm: float = DEFAULTS["precipitation_mm"]
    event_attendance_est: float = DEFAULTS["event_attendance_est"]
    traffic_congestion_index: float = DEFAULTS["traffic_congestion_index"]
    actual_departure_delay_min: float = DEFAULTS["actual_departure_delay_min"]
    holiday: int = Field(DEFAULTS["holiday"], ge=0, le=1)
    peak_hour: int = Field(DEFAULTS["peak_hour"], ge=0, le=1)
    weekday: int = Field(DEFAULTS["weekday"], ge=0, le=6)

    model_config = {
        "json_schema_extra": {
            "example": {"transport_type": "Tram", "weather_condition": "Storm",
                        "event_type": "Sports", "season": "Winter", "precipitation_mm": 13.0,
                        "traffic_congestion_index": 80, "actual_departure_delay_min": 10,
                        "peak_hour": 1}
        }
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": _art()["model_name"]}


@app.post("/predict")
def predict_endpoint(trip: Trip) -> dict:
    """Retraso de llegada estimado (minutos). Recuerda: R² ~ 0, no fiable a nivel de viaje."""
    pred = predict([trip.model_dump()], art=_art())[0]
    return {
        "predicted_arrival_delay_min": pred,
        "model": _art()["model_name"],
        "disclaimer": "R^2 ~ 0: estimacion cercana al promedio historico, no fiable a nivel de "
                      "viaje individual. Ver MODEL_CARD.md.",
    }
