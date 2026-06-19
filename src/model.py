"""Modelos, baseline y métricas de regresión."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def build_models(cfg: dict) -> dict:
    """Los tres modelos a comparar: un lineal interpretable y dos ensembles."""
    rs = cfg["seed"]
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(**cfg["models"]["random_forest"],
                                               random_state=rs, n_jobs=-1),
        "XGBoost": XGBRegressor(**cfg["models"]["xgboost"], random_state=rs, n_jobs=-1),
    }


def build_baseline() -> DummyRegressor:
    """Baseline obligatorio: predecir siempre la media del train.

    Es la vara de medir. Si los modelos "de verdad" no le ganan, es que las features no aportan
    señal — exactamente lo que pasa aquí. Sin este baseline, un R² ~ 0 no se interpreta bien.
    """
    return DummyRegressor(strategy="mean")


def metrics(y_true, y_pred) -> dict:
    """MAE y RMSE en minutos; R² (varianza explicada — negativo = peor que predecir la media)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate(model, X_tr, y_tr, X_te, y_te, cv: int = 5) -> dict:
    """Entrena, evalúa en test y añade el MAE de validación cruzada (estabilidad en dataset chico)."""
    cv_mae = -cross_val_score(model, X_tr, y_tr, cv=cv,
                              scoring="neg_mean_absolute_error").mean()
    model.fit(X_tr, y_tr)
    m = metrics(y_te, model.predict(X_te))
    return {"CV_MAE": float(cv_mae), **m}
