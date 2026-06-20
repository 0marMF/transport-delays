"""Reporte descriptivo: el producto real de este proyecto.

El modelo no predice (R² ~ 0), así que la pregunta declarada —*cómo afectan clima y eventos al
retraso*— se responde de forma **descriptiva y honesta**: cuántos minutos suma cada condición, qué
eventos pesan más, y si reformular como clasificación ("retraso severo sí/no") rescata algo de
señal. Spoiler honesto: no. Las diferencias son ruido y la clasificación da AUC ~ 0.5.

    python -m src.report          # imprime el resumen y escribe reports/insights.{json,md} + figura
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from . import data, features
from .config import load_config, path


def _delta_table(df: pd.DataFrame, group_col: str, target: str, baseline: str) -> pd.DataFrame:
    """Retraso medio por categoría y su diferencia (delta) contra una categoría de referencia."""
    g = df.groupby(group_col)[target].agg(mean="mean", n="count")
    g["delta"] = g["mean"] - g.loc[baseline, "mean"]
    return g.sort_values("delta", ascending=False)


def weather_impact(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Minutos de retraso por clima, contra 'Clear' (cielo despejado) como referencia."""
    return _delta_table(df, "weather_condition", cfg["data"]["target"], "Clear")


def event_impact(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Minutos de retraso por tipo de evento, contra 'None' (sin evento) como referencia."""
    return _delta_table(df, "event_type", cfg["data"]["target"], "None")


def classification_signal(df: pd.DataFrame, X: pd.DataFrame, cfg: dict,
                          thresholds=(15, 18, 20)) -> dict:
    """¿Y si predecimos 'retraso severo sí/no'? AUC ~ 0.5 = sin señal (igual que tirar una moneda)."""
    target = cfg["data"]["target"]
    Xs = StandardScaler().fit_transform(X)
    out = {}
    for thr in thresholds:
        yb = (df[target] >= thr).astype(int)
        auc = cross_val_score(LogisticRegression(max_iter=2000), Xs, yb, cv=5,
                              scoring="roc_auc").mean()
        out[str(thr)] = {"severe_rate": round(float(yb.mean()), 3), "auc": round(float(auc), 3)}
    return out


def build_insights(cfg: dict | None = None) -> dict:
    """Calcula todos los insights descriptivos y los devuelve como dict serializable."""
    cfg = cfg or load_config()
    df = data.load(cfg)
    X, _ = features.build_design_matrix(df, cfg)
    target = cfg["data"]["target"]

    w, e = weather_impact(df, cfg), event_impact(df, cfg)
    clf = classification_signal(df, X, cfg)
    std = float(df[target].std())
    max_w, max_e = float(w["delta"].abs().max()), float(e["delta"].abs().max())
    max_auc_gap = max(abs(v["auc"] - 0.5) for v in clf.values())

    # Veredicto honesto, derivado de los números (no escrito a mano).
    sin_senal = max_w < 1 and max_e < 1 and max_auc_gap < 0.05
    verdict = (
        "Sin señal utilizable. Las diferencias por clima (<{:.2f} min) y por evento (<{:.2f} min) "
        "son ruido frente a una desviación de {:.1f} min, y reformular como clasificación da "
        "AUC ~ 0.5. Clima y eventos de este dataset no explican el retraso."
    ).format(max_w, max_e, std) if sin_senal else (
        "Hay algún efecto descriptivo; revisar magnitudes antes de concluir."
    )

    return {
        "global_mean_delay_min": round(float(df[target].mean()), 2),
        "global_std_delay_min": round(std, 2),
        "weather_delta_vs_clear": {k: {"mean": round(r["mean"], 2), "delta": round(r["delta"], 2),
                                       "n": int(r["n"])} for k, r in w.iterrows()},
        "event_delta_vs_none": {k: {"mean": round(r["mean"], 2), "delta": round(r["delta"], 2),
                                    "n": int(r["n"])} for k, r in e.iterrows()},
        "max_abs_weather_delta_min": round(max_w, 2),
        "max_abs_event_delta_min": round(max_e, 2),
        "classification_signal": clf,
        "verdict": verdict,
    }


def _to_markdown(ins: dict) -> str:
    """Convierte los insights en un reporte legible (lo que leería un operador o un reclutador)."""
    lines = ["# Reporte descriptivo — retrasos de transporte", "",
             f"Retraso medio global: **{ins['global_mean_delay_min']} min** "
             f"(desviación {ins['global_std_delay_min']} min).", "",
             "## Minutos extra por clima (vs. cielo despejado)", "",
             "| Clima | Retraso medio (min) | Delta vs Clear | n |", "|---|---|---|---|"]
    for k, r in ins["weather_delta_vs_clear"].items():
        lines.append(f"| {k} | {r['mean']} | {r['delta']:+} | {r['n']} |")
    lines += ["", "## Minutos extra por evento (vs. sin evento)", "",
              "| Evento | Retraso medio (min) | Delta vs None | n |", "|---|---|---|---|"]
    for k, r in ins["event_delta_vs_none"].items():
        lines.append(f"| {k} | {r['mean']} | {r['delta']:+} | {r['n']} |")
    lines += ["", "## ¿Y como clasificación (retraso severo sí/no)?", "",
              "| Umbral (min) | % severos | AUC (CV) |", "|---|---|---|"]
    for thr, v in ins["classification_signal"].items():
        lines.append(f"| >= {thr} | {v['severe_rate']:.1%} | {v['auc']} |")
    lines += ["", "## Veredicto", "", ins["verdict"], "",
              "## Recomendaciones para operadores", "",
              "- **No usar clima ni eventos de este dataset para anticipar retrasos:** no tienen "
              "poder descriptivo (diferencias < 1 min, dentro del ruido) ni predictivo (R² ~ 0, "
              "AUC ~ 0.5).",
              "- **Planificar con el promedio + margen:** el retraso medio (~13 min) es estable; "
              "operativamente es más sensato asumir ese promedio que confiar en 'predicciones' de "
              "estas variables.",
              "- **Para un modelo útil hacen falta datos con señal real:** GPS/AVL en tiempo real, "
              "ocupación, incidencias/averías, obras, headway y el retraso de salida observado.", ""]
    return "\n".join(lines)


def save_figure(ins: dict, cfg: dict) -> None:
    """Figura de los deltas vs. la desviación global — para que se vea que son ruido."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    std = ins["global_std_delay_min"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, (titulo, key, ref) in zip(ax, [("Clima (vs Clear)", "weather_delta_vs_clear", "Clear"),
                                          ("Evento (vs None)", "event_delta_vs_none", "None")]):
        d = {k: v["delta"] for k, v in ins[key].items() if k != ref}
        a.barh(list(d), list(d.values()), color="#3498db")
        a.axvline(0, color="k", lw=.8)
        a.axvspan(-std, std, color="gray", alpha=.12, label=f"±1 desv. ({std:.1f} min)")
        a.set_title(f"Delta de retraso — {titulo}")
        a.set_xlabel("minutos sobre la referencia")
        a.set_xlim(-std * 1.1, std * 1.1)   # misma escala que el ruido: los deltas casi no se ven
        a.legend(loc="lower right")
    fig.suptitle("Clima y eventos apenas mueven el retraso (los deltas caben dentro del ruido)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path(cfg["paths"]["reports"]) / "09_descriptive_impact.png", bbox_inches="tight")
    plt.close(fig)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ins = build_insights(cfg)
    reports = path(cfg["paths"]["reports"])
    reports.mkdir(exist_ok=True)
    with open(reports / "insights.json", "w", encoding="utf-8") as f:
        json.dump(ins, f, indent=2, ensure_ascii=False)
    with open(reports / "insights.md", "w", encoding="utf-8") as f:
        f.write(_to_markdown(ins))
    save_figure(ins, cfg)

    print(f"Retraso medio: {ins['global_mean_delay_min']} min (desv {ins['global_std_delay_min']})")
    print(f"Mayor delta por clima:  {ins['max_abs_weather_delta_min']} min")
    print(f"Mayor delta por evento: {ins['max_abs_event_delta_min']} min")
    print(f"Clasificación severo sí/no: AUC ~ {[v['auc'] for v in ins['classification_signal'].values()]}")
    print("Veredicto:", ins["verdict"])
    return ins


if __name__ == "__main__":
    run()
