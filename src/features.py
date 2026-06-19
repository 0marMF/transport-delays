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


def build_design_matrix(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) listos para modelar.

    Quita identificadores, columnas de fecha/hora crudas y las de leakage; codifica las categóricas
    con One-Hot (drop_first para evitar colinealidad) y pasa las banderas booleanas a 0/1.
    """
    cfg = cfg or load_config()
    target = cfg["data"]["target"]
    feats = cfg["features"]

    df = add_engineered(df, cfg)
    y = df[target]

    drop = [*feats["id_cols"], *feats["leakage_cols"], target]
    X = df.drop(columns=[c for c in drop if c in df.columns])

    cat = X.select_dtypes("object").columns.tolist()
    X = pd.get_dummies(X, columns=cat, drop_first=True)
    X = X.astype({c: "int" for c in X.select_dtypes("bool").columns})
    return X, y
