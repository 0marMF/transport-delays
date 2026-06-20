"""Tests del reporte descriptivo (sub-funciones, sobre datos sintéticos)."""
from src import data, features, report


def test_impacto_clima_y_evento(synth_df, cfg):
    df = data.clean(synth_df)
    w = report.weather_impact(df, cfg)
    e = report.event_impact(df, cfg)
    # La referencia tiene delta 0 por construcción.
    assert w.loc["Clear", "delta"] == 0
    assert e.loc["None", "delta"] == 0
    assert "mean" in w.columns and "n" in w.columns


def test_classification_signal(synth_df, cfg):
    df = data.clean(synth_df)
    X, _ = features.build_design_matrix(df, cfg)
    sig = report.classification_signal(df, X, cfg, thresholds=(15,))
    assert "15" in sig
    assert 0.0 <= sig["15"]["auc"] <= 1.0
    assert 0.0 <= sig["15"]["severe_rate"] <= 1.0
