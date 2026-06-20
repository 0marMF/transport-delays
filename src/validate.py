"""Validación de datos (data contract) y guard anti-leakage.

El verdadero entregable de Ciencia de Datos de este proyecto. Como el modelo no predice (R² ~ 0),
el valor está en un pipeline donde los errores de datos (rangos imposibles, categorías raras) y
—sobre todo— el *data leakage* sean **imposibles de colar** sin que el pipeline grite. Todo es
fail-fast: se valida ANTES de entrenar.
"""
import pandas as pd

from .config import load_config


class DataContractError(ValueError):
    """El dataset crudo viola alguna regla del contrato de datos."""


class LeakageError(ValueError):
    """Una columna derivada del target (fuga de información) llegó a la matriz de features."""


def check_contract(df: pd.DataFrame, cfg: dict | None = None) -> list[str]:
    """Revisa el contrato y devuelve la lista de violaciones (vacía si todo está bien)."""
    cfg = cfg or load_config()
    c = cfg["contract"]
    issues: list[str] = []

    missing = [col for col in c["required_cols"] if col not in df.columns]
    if missing:
        issues.append(f"Faltan columnas requeridas: {missing}")

    nullable = set(c.get("nullable", []))
    for col in df.columns:
        if col not in nullable and df[col].isna().any():
            issues.append(f"'{col}' tiene {int(df[col].isna().sum())} nulos y no es nullable")

    for col, (lo, hi) in c.get("ranges", {}).items():
        if col in df.columns:
            n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_bad:
                issues.append(f"'{col}' fuera de rango [{lo}, {hi}] en {n_bad} filas")

    for col in c.get("binary", []):
        if col in df.columns:
            extra = set(df[col].dropna().unique()) - {0, 1}
            if extra:
                issues.append(f"'{col}' deberia ser binaria 0/1; valores extra: {sorted(extra)}")

    for col, allowed in c.get("categories", {}).items():
        if col in df.columns:
            extra = set(df[col].dropna().unique()) - set(allowed)
            if extra:
                issues.append(f"'{col}' con categorias no esperadas: {sorted(extra)}")

    return issues


def validate(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Lanza DataContractError si el dataset viola el contrato; si no, devuelve df (encadenable)."""
    issues = check_contract(df, cfg)
    if issues:
        raise DataContractError("El dataset viola el contrato de datos:\n- " + "\n- ".join(issues))
    return df


def assert_no_leakage(X: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Garantiza que ninguna columna de leakage (ni el target) entre como feature.

    `delayed` ES el target binarizado: si se cuela, el R² se dispara y el resultado es mentira.
    Este guard hace ese error imposible de repetir — el pipeline falla antes de entrenar. Cubre
    también las dummies que pudieran derivarse de una columna prohibida (p. ej. `delayed_1`).
    """
    cfg = cfg or load_config()
    forbidden = set(cfg["features"]["leakage_cols"]) | {cfg["data"]["target"]}
    present = [col for col in X.columns
               if col in forbidden or any(col.startswith(f"{f}_") for f in forbidden)]
    if present:
        raise LeakageError(
            f"Columnas de leakage en la matriz de features: {present}. "
            "Revisa features.build_design_matrix y config.features.leakage_cols."
        )
    return X


if __name__ == "__main__":
    # Chequeo rápido del contrato sobre el dataset crudo: `python -m src.validate`.
    import sys

    from . import data
    issues = check_contract(data.load_raw())
    if issues:
        print("Contrato de datos VIOLADO:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    print("Contrato de datos OK: el dataset cumple rangos, categorias y nulos esperados.")

