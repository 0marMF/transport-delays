"""Test del scoring (camino de inferencia) con un artefacto sintético, sin tocar el CSV real."""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src import data, features, score


def _artefacto_sintetico(synth_df, cfg):
    X, y = features.build_design_matrix(data.clean(synth_df), cfg)
    scale_cols = [c for c in cfg["features"]["scale_cols"] if c in X.columns]
    scaler = StandardScaler().fit(X[scale_cols])
    Xs = X.copy()
    Xs[scale_cols] = scaler.transform(Xs[scale_cols])
    mdl = LinearRegression().fit(Xs, y)
    return {"model": mdl, "model_name": "LR", "features": X.columns.tolist(),
            "scaler": scaler, "scale_cols": scale_cols}


def test_predict_varios_viajes(synth_df, cfg):
    art = _artefacto_sintetico(synth_df, cfg)
    # Un viaje detallado y otro vacío (todo defaults): ambos deben devolver un número.
    preds = score.predict([{"weather_condition": "Storm", "event_type": "Sports"}, {}],
                          cfg=cfg, art=art)
    assert len(preds) == 2
    assert all(isinstance(p, float) for p in preds)


def test_prepare_alinea_columnas(synth_df, cfg):
    art = _artefacto_sintetico(synth_df, cfg)
    X = score.prepare([{"weather_condition": "Storm"}], art, cfg)
    assert list(X.columns) == art["features"]   # alineado exactamente a lo que espera el modelo
