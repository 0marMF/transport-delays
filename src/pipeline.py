"""Pipeline de un comando: datos -> features -> baseline + modelos -> métricas + modelo servible.

    python -m src.pipeline

Reproduce de forma honesta el resultado del proyecto (R² ~ 0: las features no predicen el retraso)
y deja `reports/metrics.json` y `src/best_model.pkl` listos. El DummyRegressor se reporta como
referencia explícita para que el R² ~ 0 se lea en contexto.
"""
import csv
import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import data, features, model, validate
from .config import load_config, path


def _scale(scaler, frame: pd.DataFrame, cols: list[str], fit: bool) -> pd.DataFrame:
    """Escala in-place las columnas continuas (fit en train, solo transform en test)."""
    out = frame.copy()
    out[cols] = scaler.fit_transform(out[cols]) if fit else scaler.transform(out[cols])
    return out


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    rs = cfg["seed"]
    scale_cols = [c for c in cfg["features"]["scale_cols"]]

    # 1) Contrato de datos sobre el CRUDO (antes de limpiar): rangos, categorías, nulos esperados.
    df_raw = data.load_raw(cfg)
    validate.validate(df_raw, cfg)
    df = data.clean(df_raw)

    # 2) Features + guard anti-leakage: el pipeline falla aquí si `delayed` (target binarizado)
    #    se cuela como feature. Hace ese error imposible de repetir.
    X, y = features.build_design_matrix(df, cfg)
    validate.assert_no_leakage(X, cfg)
    scale_cols = [c for c in scale_cols if c in X.columns]

    # Split 80/20. La matriz de diseño se construye sobre TODO el dataset antes de partir, así las
    # columnas dummy son idénticas en train y test (ninguna categoría se queda fuera por azar).
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["split"]["test_size"],
                                              random_state=rs)
    split_scaler = StandardScaler()
    X_tr = _scale(split_scaler, X_tr, scale_cols, fit=True)
    X_te = _scale(split_scaler, X_te, scale_cols, fit=False)

    # Baseline (predecir la media) + los tres modelos, todos con CV de 5 folds.
    baseline = model.evaluate(model.build_baseline(), X_tr, y_tr, X_te, y_te)
    results = {name: model.evaluate(m, X_tr, y_tr, X_te, y_te)
               for name, m in model.build_models(cfg).items()}

    best = min(results, key=lambda k: results[k]["MAE"])

    metrics = {
        "best_model": best,
        "baseline": {"DummyRegressor (media)": {k: round(v, 3) for k, v in baseline.items()}},
        "comparison": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in results.items()},
    }
    path(cfg["paths"]["reports"]).mkdir(exist_ok=True)
    with open(path(cfg["paths"]["metrics"]), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _save_served_model(X, y, scale_cols, best, cfg)
    _log_experiment(metrics, cfg)

    print(f"Mejor modelo (menor MAE): {best}  |  R2={results[best]['R2']:.3f}  MAE={results[best]['MAE']:.2f} min")
    print(f"Baseline DummyRegressor: MAE={baseline['MAE']:.2f} min  R2={baseline['R2']:.3f}")
    print("Lectura honesta: los modelos apenas le ganan al baseline -> las features no tienen señal.")
    return metrics


def _save_served_model(X, y, scale_cols, best_name, cfg):
    """Reentrena el mejor modelo con TODO el dataset (más datos = mejor) y lo serializa para la API."""
    scaler = StandardScaler()
    X_full = X.copy()
    X_full[scale_cols] = scaler.fit_transform(X_full[scale_cols])
    mdl = model.build_models(cfg)[best_name]
    mdl.fit(X_full, y)
    with open(path(cfg["data"]["model"]), "wb") as f:
        pickle.dump({"model": mdl, "model_name": best_name, "features": X.columns.tolist(),
                     "scaler": scaler, "scale_cols": scale_cols}, f)


def _log_experiment(metrics, cfg):
    """Añade una fila a experiments.csv para poder comparar corridas en el tiempo."""
    fp = path(cfg["paths"]["experiments"])
    row = {"best_model": metrics["best_model"],
           **{f"{name}_{k}": v for name, m in metrics["comparison"].items() for k, v in m.items()}}
    write_header = not fp.exists()
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    run()
