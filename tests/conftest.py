"""Fixtures compartidas.

Los tests NO usan el CSV real (no se versiona): trabajan sobre un DataFrame sintético que imita el
esquema y cumple el contrato de datos. Así corren en cualquier sitio, incluida la CI.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture
def cfg():
    """La config real del proyecto (incluye el contrato de datos)."""
    return load_config()


@pytest.fixture
def synth_df():
    """Dataset crudo sintético (200 filas) válido frente al contrato.

    Incluye nulos en `event_type` (= "no hubo evento"), que el contrato permite. El target es ruido:
    el proyecto justamente trata de un dataset sin señal, así que no hace falta simular relación.
    """
    rng = np.random.RandomState(0)
    n = 200
    return pd.DataFrame({
        "trip_id": [f"T{i:05d}" for i in range(n)],
        "date": "2023-01-01",
        "time": "05:00:00",
        "transport_type": rng.choice(["Bus", "Metro", "Train", "Tram"], n),
        "route_id": "Route_1",
        "origin_station": "Station_1",
        "destination_station": "Station_2",
        "scheduled_departure": "05:00:00",
        "scheduled_arrival": "05:30:00",
        "actual_departure_delay_min": rng.randint(0, 20, n),
        "actual_arrival_delay_min": rng.randint(0, 30, n),
        "weather_condition": rng.choice(["Clear", "Cloudy", "Fog", "Rain", "Snow", "Storm"], n),
        "temperature_C": rng.uniform(-5, 35, n),
        "humidity_percent": rng.uniform(30, 99, n),
        "wind_speed_kmh": rng.uniform(0, 59, n),
        "precipitation_mm": rng.uniform(0, 20, n),
        "event_type": rng.choice(["Concert", "Festival", "Parade", "Protest", "Sports", None], n),
        "event_attendance_est": rng.randint(0, 50000, n),
        "traffic_congestion_index": rng.uniform(0, 99, n),
        "holiday": rng.randint(0, 2, n),
        "peak_hour": rng.randint(0, 2, n),
        "weekday": rng.randint(0, 7, n),
        "season": rng.choice(["Winter", "Spring", "Summer", "Autumn"], n),
        "delayed": rng.randint(0, 2, n),
    })
