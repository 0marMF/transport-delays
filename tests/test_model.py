"""Tests de métricas, baseline y evaluación."""
from sklearn.model_selection import train_test_split

from src import data, features, model


def test_metrics_prediccion_perfecta():
    m = model.metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1


def test_evaluate_devuelve_claves(synth_df, cfg):
    X, y = features.build_design_matrix(data.clean(synth_df), cfg)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    res = model.evaluate(model.build_baseline(), X_tr, y_tr, X_te, y_te, cv=3)
    assert {"CV_MAE", "MAE", "RMSE", "R2"} <= set(res)


def test_baseline_predice_la_media(synth_df, cfg):
    X, y = features.build_design_matrix(data.clean(synth_df), cfg)
    base = model.build_baseline().fit(X, y)
    # DummyRegressor(strategy="mean") predice siempre la media del train.
    assert abs(base.predict(X.head(3))[0] - y.mean()) < 1e-6
