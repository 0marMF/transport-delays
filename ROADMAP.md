# Roadmap — Public Transport Delays Prediction

**Proyecto de portfolio:** Omar Mora Flores  
**Objetivo:** Predecir retrasos en transporte público usando factores externos como clima y eventos de ciudad, demostrando habilidades en regresión, feature engineering y análisis temporal.

**Dataset:** [Public Transport Delays with Weather and Events — Kaggle](https://www.kaggle.com/datasets/khushikyad001/public-transport-delays-with-weather-and-events)

---

## Estado general

| Fase | Componente | Estado |
|---|---|---|
| 0 | Setup del entorno | ✅ Completado |
| 1 | EDA | ✅ Completado |
| 2 | Preprocessing & Feature Engineering | ✅ Completado |
| 3 | Modelado & Evaluación | ✅ Completado |
| 4 | Cierre de portfolio | ✅ Completado |

Leyenda: ⬜ Pendiente · 🔄 En progreso · ✅ Completado

---

## Fase 0 — Setup del entorno

- [ ] Descargar dataset de Kaggle y colocar en `data/`
- [ ] Crear `requirements.txt` con: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, plotly
- [ ] Crear `.gitignore` (excluir `data/`, `*.csv`, `*.pkl`, `__pycache__/`)
- [ ] Cargar dataset y confirmar que se lee correctamente

### Entregables
- `requirements.txt` funcional
- Dataset cargado y explorado brevemente

---

## Fase 1 — EDA

**Archivo:** `notebooks/01_EDA.ipynb`  
**Pregunta central:** ¿Cómo afectan el clima y los eventos urbanos a los retrasos del transporte?

### Secciones

#### 1.1 Carga y descripción del dataset
- [ ] Shape, dtypes, valores nulos, estadísticas descriptivas
- [ ] Identificar columnas clave: tiempo de retraso, condición climática, tipo de evento, ruta/línea

#### 1.2 Distribución de la variable objetivo (delay)
- [ ] Histograma de duración de retrasos
- [ ] Porcentaje de viajes sin retraso vs. con retraso
- [ ] Guardar → `reports/01_delay_distribution.png`

#### 1.3 Retrasos por condición climática
- [ ] Retraso promedio por tipo de clima (lluvia, nieve, temperatura extrema, normal)
- [ ] Boxplot de retrasos agrupado por clima
- [ ] Guardar → `reports/02_weather_impact.png`

#### 1.4 Retrasos por eventos de ciudad
- [ ] Comparar retrasos en días con eventos vs. días normales
- [ ] Top 5 tipos de eventos con mayor impacto en retrasos
- [ ] Guardar → `reports/03_events_impact.png`

#### 1.5 Análisis temporal
- [ ] Retrasos por hora del día y día de la semana
- [ ] Heatmap hora × día con intensidad de retraso promedio
- [ ] Guardar → `reports/04_temporal_heatmap.png`

#### 1.6 Correlación de features con el retraso
- [ ] Matriz de correlación de variables numéricas con delay
- [ ] Guardar → `reports/05_correlation.png`

#### 1.7 Conclusiones del EDA
- [ ] Markdown con hallazgos que justifican las decisiones de feature engineering

### Entregables
- `notebooks/01_EDA.ipynb` sin errores
- 5 imágenes en `reports/`

---

## Fase 2 — Preprocessing & Feature Engineering

**Archivo:** `notebooks/02_preprocessing.ipynb`

### Tareas

#### 2.1 Limpieza
- [ ] Manejar valores nulos (imputación o eliminación según cantidad)
- [ ] Eliminar duplicados
- [ ] Corregir tipos de datos (fechas, categorías)

#### 2.2 Feature Engineering
- [ ] Extraer `hour`, `day_of_week`, `month`, `is_weekend` desde columna de fecha/hora
- [ ] Codificar variables categóricas: clima, tipo de evento, línea de transporte (One-Hot o Label Encoding)
- [ ] Crear `has_event` — bandera binaria si hubo evento ese día
- [ ] Crear `extreme_weather` — bandera si temperatura fuera de rango normal

#### 2.3 Escalado
- [ ] `StandardScaler` o `RobustScaler` para features numéricas continuas

#### 2.4 Split train/test
- [ ] Split 80/20 con `random_state=42`
- [ ] Si hay datos temporales ordenados: usar split cronológico (no aleatorio)

#### 2.5 Guardar splits
- [ ] Serializar en `data/splits.pkl`

### Entregables
- `notebooks/02_preprocessing.ipynb` sin errores
- `data/splits.pkl`

---

## Fase 3 — Modelado & Evaluación

**Archivo:** `notebooks/03_modeling.ipynb`  
**Tipo de problema:** Regresión (predicción de duración de retraso en minutos)

### Modelos a comparar

| Modelo | Propósito |
|---|---|
| Linear Regression | Baseline interpretable |
| Random Forest Regressor | Ensemble robusto |
| XGBoost Regressor | Modelo principal |

### Métricas
- MAE (Mean Absolute Error) — error promedio en minutos
- RMSE (Root Mean Squared Error) — penaliza errores grandes
- R² — varianza explicada por el modelo

### Secciones

#### 3.1 Entrenamiento de los 3 modelos
- [ ] Entrenar, predecir y calcular métricas para cada modelo
- [ ] Tabla comparativa de resultados

#### 3.2 Visualizaciones
- [ ] Gráfico de barras de MAE/RMSE por modelo → `reports/06_model_comparison.png`
- [ ] Scatter plot: retraso real vs. predicho (mejor modelo) → `reports/07_predictions.png`
- [ ] Feature importance del mejor modelo → `reports/08_feature_importance.png`

#### 3.3 Análisis de errores
- [ ] ¿El modelo falla más con ciertos tipos de clima o eventos?
- [ ] Distribución de residuos

#### 3.4 Serialización
- [ ] Guardar mejor modelo en `src/best_model.pkl`

#### 3.5 Interpretación de negocio
- [ ] ¿Cuántos minutos de retraso agrega la lluvia en promedio?
- [ ] ¿Qué eventos tienen mayor impacto?
- [ ] Recomendaciones para operadores de transporte

### Entregables
- `notebooks/03_modeling.ipynb` sin errores
- `src/best_model.pkl`
- 3 imágenes en `reports/`

---

## Fase 4 — Cierre de portfolio

- [ ] Ejecutar pipeline completo sin errores en entorno limpio
- [ ] Escribir `README.md` con: descripción, hallazgos clave, instrucciones de ejecución, stack técnico
- [ ] Incluir al menos 3 gráficos en el README
- [ ] Verificar que `.gitignore` excluye el dataset

### Checklist de calidad
- [ ] Narrativa en Markdown entre celdas de código
- [ ] Gráficos con títulos, ejes etiquetados y leyendas
- [ ] Sin rutas absolutas hardcodeadas
- [ ] `random_state=42` en todos los pasos

---

## Orden de desarrollo

```
Fase 0 → Fase 1 → Fase 2 → Fase 3 → Fase 4
  Setup    EDA    Preproc   Model    Cierre
```

---

## Notas técnicas

- Si el dataset tiene columna de fecha: usar split cronológico en train/test para evitar data leakage temporal
- El problema puede tratarse como regresión (minutos de retraso) o clasificación (retraso sí/no) — elegir según lo que permita el dataset
- Documentar la decisión de split cronológico vs. aleatorio en el notebook
