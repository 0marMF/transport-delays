# 🚆 Public Transport Delays Prediction

> **Predecir retrasos de transporte con clima y eventos — y un hallazgo honesto sobre los datos**
> *Pipeline de regresión completo + análisis crítico de la señal predictiva*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5+-F7931E?logo=scikit-learn)](https://scikit-learn.org)

---

## 📌 Objetivo

Predecir el **retraso de llegada (minutos)** del transporte público a partir de clima, eventos y
variables temporales (2,000 viajes), y evaluar **cuánta señal predictiva** ofrecen realmente esas
variables.

---

## 🎯 Resultados (regresión, validación cruzada 5-fold + test)

| Modelo | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| **Linear Regression** ✅ | **7.73** | 9.25 | −0.02 |
| Random Forest | 7.76 | 9.25 | −0.02 |
| XGBoost | 7.88 | 9.41 | −0.05 |

> **Hallazgo central (honesto):** el **R² ≈ 0** en todos los modelos — las features disponibles
> (clima/eventos de este dataset sintético) **no explican** el retraso. Que un modelo lineal
> empate con XGBoost confirma que **el problema es la señal de los datos, no el algoritmo**.
> Ningún modelo rescata features sin información.

---

## 🔬 Metodología

1. **EDA** (`01_EDA.ipynb`) — distribución del retraso, impacto de clima y eventos, patrones
   temporales, correlaciones.
2. **Preprocessing** (`02_preprocessing.ipynb`) — imputación de `event_type`, feature engineering
   (`has_event`, `extreme_weather`), One-Hot encoding, escalado. **Se excluye `delayed`** porque es
   el target binarizado (*data leakage*).
3. **Modelado** (`03_modeling.ipynb`) — Linear Regression, Random Forest y XGBoost con
   **validación cruzada 5-fold** (dataset pequeño), métricas MAE/RMSE/R² y análisis de errores.

---

## ⚠️ Lecciones del proyecto

- **Detectar y evitar *data leakage*:** `delayed` correlaciona 0.76 con el target porque *es* el
  target binarizado — incluirlo habría inflado falsamente el rendimiento.
- **Un R² bajo bien diagnosticado vale más que un número inflado:** el valor del análisis está en
  demostrar, con validación cruzada, que las features no tienen señal.
- **Dataset pequeño (2,000) → validación cruzada** obligatoria para no confiar en un único split.

---

## 🏗️ Estructura

```
transport-delays/
├── data/                         # dataset (no versionado) + splits.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/best_model.pkl
├── reports/                      # 8 visualizaciones + metrics.json
├── HALLAZGOS.md
├── README.md
└── ROADMAP.md
```

---

## 🚀 Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb
```

> Dataset: [Public Transport Delays with Weather and Events — Kaggle](https://www.kaggle.com/datasets/khushikyad001/public-transport-delays-with-weather-and-events)
> (colócalo en `data/`; no se versiona).

> 📄 Detalle de detecciones y aprendizajes en [`HALLAZGOS.md`](HALLAZGOS.md).

---

## 👨‍💻 Autor

**Omar Mora Flores** · Data Analyst & ML Engineer
📧 omar13mor@gmail.com · 🔗 [linkedin.com/in/omar-mora-flores](https://linkedin.com/in/omar-mora-flores)
