# Model Card — Predicción de retrasos de transporte público

Ficha del modelo que estima el retraso de llegada del transporte público. Una página para dejar
claro qué hace, con qué datos, cómo se evaluó y —el punto más importante de este proyecto— **por
qué NO debe usarse para decisiones**.

## Resumen

- **Tarea:** regresión del retraso de llegada (`actual_arrival_delay_min`, en minutos) a partir de
  clima, eventos, tráfico y variables temporales.
- **Modelo servido:** Linear Regression (gana por MAE, pero ver abajo: **ningún modelo funciona**).
- **Conclusión honesta:** **R² ≈ 0.** Los modelos no le ganan a predecir la media. Las features no
  tienen señal sobre el retraso. Este model card existe, sobre todo, para documentar ese límite.
- **Versión:** 1.1.0 · **Artefacto:** `src/best_model.pkl` (modelo + scaler + nombres de feature).

## Para qué sirve (y para qué no)

- **Uso previsto:** demostración de portfolio — un pipeline de regresión reproducible y, sobre todo,
  un **diagnóstico honesto** de que estas variables no predicen el retraso.
- **Fuera de alcance — NO usar para decisiones:** no sirve para anticipar el retraso de un viaje
  concreto, ni para planificar operaciones, ni para informar a pasajeros. Con R² ≈ 0, la API
  devuelve esencialmente el promedio histórico (~13 min) sin importar el input.

## Datos

- Public Transport Delays with Weather and Events (Kaggle): **2,000 viajes** con clima, eventos,
  tráfico y variables temporales. El dataset **parece sintético** (ver más abajo) y **no se versiona**.
- Limpieza: los ~1,173 nulos de `event_type` significan "sin evento" → se rellenan con `None`.
- **Anti-leakage:** se excluye `delayed` porque **es el target binarizado** (correlación 0.76 con el
  retraso por construcción). Un guard automático (`src/validate.py`) hace ese error imposible.
- Split 80/20 con `random_state=42` y validación cruzada de 5 folds (dataset pequeño).

## Evaluación

Ningún modelo supera al baseline que solo predice la media:

| Modelo | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| DummyRegressor (media) | 7.70 | 9.19 | −0.00 |
| Linear Regression | 7.73 | 9.25 | −0.02 |
| Random Forest | 7.76 | 9.25 | −0.02 |
| XGBoost | 7.88 | 9.41 | −0.05 |

A nivel **descriptivo** (`reports/insights.md`) el resultado es el mismo: la mayor diferencia de
retraso por clima es 0.62 min y por evento 0.76 min —ruido frente a una desviación de 9.3 min, y
con signos físicamente absurdos (tormentas/eventos *reduciendo* el retraso)—. Reformular como
clasificación ("retraso severo sí/no") da **AUC ≈ 0.5**. No hay señal ni para predecir ni para
describir.

## Por qué pasa esto (y cómo se interpreta)

- **El dataset es casi con seguridad sintético/aleatorio:** ni siquiera `actual_departure_delay_min`
  (el retraso de salida, que en el mundo real predice fuertemente el de llegada) aporta señal. Eso
  delata datos generados al azar, no un fenómeno real mal modelado.
- **No es un fallo del pipeline:** que un modelo lineal, Random Forest y XGBoost empaten entre sí y
  con la media confirma que el techo lo ponen los datos, no el algoritmo.

## Límites y consideraciones

- **No usar para ninguna decisión operativa.** El modelo no predice.
- **Faltan los drivers reales:** un modelo útil necesitaría temperatura real ligada a la ruta,
  GPS/AVL en tiempo real, ocupación, incidencias/averías, obras y headway.
- **Dataset pequeño (2,000) y sintético:** conclusiones de negocio limitadas a "estos datos no
  sirven para esto".

## Monitoreo y mantenimiento

- `reports/experiments.csv` registra cada corrida del pipeline para comparar en el tiempo.
- Si se consiguen **datos reales con señal**, reentrenar (`python -m src.pipeline`) y re-evaluar:
  el guard anti-leakage y el contrato de datos (`python -m src.validate`) siguen aplicando.

## Cómo usarlo

- Pipeline: `python -m src.pipeline` · Reporte descriptivo: `python -m src.report`.
- Scoring (demostración de serving, no fiable): `python -m src.score ...` o API `uvicorn src.api:app`.

---
*Autor: Omar Mora Flores · Última actualización: 2026-06-20*
