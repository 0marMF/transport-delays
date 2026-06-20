"""Scoring de un viaje: del input crudo a un retraso de llegada estimado (minutos).

Una sola fuente de verdad para puntuar, que usan tanto la API (`src/api.py`) como el CLI. Recibe
los atributos de un viaje (clima, evento, tráfico, retraso de salida...) y devuelve el retraso de
llegada estimado.

Nota honesta: el modelo tiene R² ~ 0 (las features no predicen el retraso), así que devolverá algo
muy cercano al promedio histórico (~13 min) casi sin importar el input. La API existe para demostrar
el serving del pipeline, no porque el modelo sea fiable — ver MODEL_CARD.md.

    python -m src.score --weather Storm --event Sports --departure-delay 10
"""
import pickle

import pandas as pd

from . import data, features
from .config import load_config, path

# Valores por defecto para los campos que no se envíen (input parcial -> viaje "neutro").
DEFAULTS = {
    "transport_type": "Bus", "weather_condition": "Clear", "event_type": "None",
    "season": "Summer", "temperature_C": 20.0, "humidity_percent": 60.0,
    "wind_speed_kmh": 10.0, "precipitation_mm": 0.0, "event_attendance_est": 0.0,
    "traffic_congestion_index": 50.0, "actual_departure_delay_min": 0.0,
    "holiday": 0, "peak_hour": 0, "weekday": 0,
}


def load_artifact(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    with open(path(cfg["data"]["model"]), "rb") as f:
        return pickle.load(f)


def prepare(records: list[dict], art: dict, cfg: dict) -> pd.DataFrame:
    """De una lista de viajes (dicts) a la matriz escalada que el modelo espera."""
    df = pd.DataFrame([{**DEFAULTS, **r} for r in records])
    df = data.clean(df)
    X = features.encode(df, cfg, drop_first=False)        # sin drop_first: ver features.encode
    X = features.align_to(X, art["features"]).astype(float)
    X[art["scale_cols"]] = art["scaler"].transform(X[art["scale_cols"]])
    return X


def predict(records: list[dict], cfg: dict | None = None, art: dict | None = None) -> list[float]:
    """Retraso de llegada estimado (min) para cada viaje."""
    cfg = cfg or load_config()
    art = art or load_artifact(cfg)
    preds = art["model"].predict(prepare(records, art, cfg))
    return [round(float(p), 2) for p in preds]


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Estima el retraso de llegada de un viaje (minutos).")
    p.add_argument("--transport-type", default="Bus")
    p.add_argument("--weather", dest="weather_condition", default="Clear")
    p.add_argument("--event", dest="event_type", default="None")
    p.add_argument("--season", default="Summer")
    p.add_argument("--temperature", dest="temperature_C", type=float, default=20.0)
    p.add_argument("--precipitation", dest="precipitation_mm", type=float, default=0.0)
    p.add_argument("--traffic", dest="traffic_congestion_index", type=float, default=50.0)
    p.add_argument("--departure-delay", dest="actual_departure_delay_min", type=float, default=0.0)
    p.add_argument("--peak-hour", dest="peak_hour", type=int, choices=[0, 1], default=0)
    trip = vars(p.parse_args())

    art = load_artifact()
    pred = predict([trip], art=art)[0]
    print(f"Modelo: {art['model_name']}")
    print(f"Retraso de llegada estimado: {pred} min")
    print("Recordatorio honesto: R² ~ 0 — esta estimación es ~el promedio historico, no fiable a "
          "nivel de viaje individual (ver MODEL_CARD.md).")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _cli()
