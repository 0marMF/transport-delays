"""Ingeniería de features y matriz de diseño.

Dos features derivadas (`has_event`, `extreme_weather`) y One-Hot de las categóricas. El detalle
crítico de este proyecto: se excluye `delayed` porque ES el target binarizado — incluirlo sería
fuga de información (data leakage) y dispararía el R² de forma engañosa. El guard automático que
hace ese error imposible vive en `validate.py` (CP2).
"""
import pandas as pd

from .config import load_config


def add_engineered(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Añade `has_event` (hubo evento ese día) y `extreme_weather` (clima severo)."""
    df = df.copy()
    df["has_event"] = (df["event_type"] != "None").astype(int)
    df["extreme_weather"] = df["weather_condition"].isin(cfg["features"]["extreme_weather"]).astype(int)
    return df


def encode(df: pd.DataFrame, cfg: dict, drop_first: bool = True) -> pd.DataFrame:
    """Matriz de features (sin target): quita ids/leakage, deriva banderas y aplica One-Hot.

    `drop_first=True` en entrenamiento evita colinealidad. En inferencia de una sola fila se usa
    `drop_first=False` y luego se alinea a las columnas entrenadas (ver `align_to`): si no, una fila
    con una sola categoría perdería su dummy.
    """
    feats = cfg["features"]
    df = add_engineered(df, cfg)
    drop = [*feats["id_cols"], *feats["leakage_cols"], cfg["data"]["target"]]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    cat = X.select_dtypes("object").columns.tolist()
    X = pd.get_dummies(X, columns=cat, drop_first=drop_first)
    return X.astype({c: "int" for c in X.select_dtypes("bool").columns})


def build_design_matrix(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) listos para entrenar."""
    cfg = cfg or load_config()
    y = df[cfg["data"]["target"]]
    return encode(df, cfg, drop_first=True), y


def align_to(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Alinea X a las columnas que el modelo espera (rellena con 0 las dummies ausentes)."""
    return X.reindex(columns=feature_names, fill_value=0)
