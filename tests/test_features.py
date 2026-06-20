"""Tests de ingeniería de features y de la matriz de diseño."""
import pandas as pd

from src import data, features


def test_encode_sin_target_ni_leakage(synth_df, cfg):
    X = features.encode(data.clean(synth_df), cfg)
    assert cfg["data"]["target"] not in X.columns
    for col in cfg["features"]["leakage_cols"]:
        assert col not in X.columns


def test_banderas_derivadas(cfg):
    df = pd.DataFrame({"event_type": ["None", "Sports"], "weather_condition": ["Clear", "Storm"]})
    out = features.add_engineered(df, cfg)
    assert out["has_event"].tolist() == [0, 1]
    assert out["extreme_weather"].tolist() == [0, 1]   # Storm es clima extremo, Clear no


def test_align_to_rellena_dummies_ausentes(cfg):
    X = pd.DataFrame({"a": [1]})
    aligned = features.align_to(X, ["a", "b", "c"])
    assert list(aligned.columns) == ["a", "b", "c"]
    assert aligned.loc[0, "b"] == 0 and aligned.loc[0, "c"] == 0


def test_inferencia_alineada_a_entrenamiento(synth_df, cfg):
    # Una sola fila (drop_first=False) + align_to debe dar exactamente las columnas entrenadas.
    X_train, _ = features.build_design_matrix(data.clean(synth_df), cfg)
    fila = features.encode(data.clean(synth_df).head(1), cfg, drop_first=False)
    alineada = features.align_to(fila, X_train.columns.tolist())
    assert list(alineada.columns) == list(X_train.columns)
