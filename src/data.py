"""Carga y limpieza del dataset de viajes."""
import pandas as pd

from .config import load_config, path


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    """Lee el CSV crudo tal cual (sin tocar columnas)."""
    cfg = cfg or load_config()
    return pd.read_csv(path(cfg["data"]["raw_csv"]))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima y honesta.

    Lo único que necesita el dataset: los ~1,173 nulos de `event_type` no son datos perdidos,
    significan "ese día no hubo evento" → los rellenamos con "None" (una categoría más), no los
    imputamos ni los tiramos.
    """
    df = df.copy()
    df["event_type"] = df["event_type"].fillna("None")
    return df


def load(cfg: dict | None = None) -> pd.DataFrame:
    """Atajo: carga + limpieza en un paso."""
    cfg = cfg or load_config()
    return clean(load_raw(cfg))
