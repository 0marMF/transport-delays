"""Test de la API. Usa el modelo versionado (src/best_model.pkl), no necesita el dataset."""
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json={"weather_condition": "Storm", "event_type": "Sports",
                                      "actual_departure_delay_min": 10, "peak_hour": 1})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["predicted_arrival_delay_min"], (int, float))
    assert "disclaimer" in body            # la honestidad va en la respuesta


def test_predict_defaults():
    # Sin campos: todo defaults. Debe responder igual (el modelo ignora el input, R^2 ~ 0).
    r = client.post("/predict", json={})
    assert r.status_code == 200
