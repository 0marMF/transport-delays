"""Carga de configuración y resolución de rutas (relativas a la raíz del proyecto)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str = "config.yaml") -> dict:
    with open(ROOT / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(relative: str) -> Path:
    return ROOT / relative
