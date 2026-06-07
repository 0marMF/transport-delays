# 🔎 Hallazgos y Aprendizajes — Public Transport Delays

> Detecciones del EDA y del modelado, más los aprendizajes del proyecto.

**Autor:** Omar Mora Flores · **Última actualización:** 2026-06-06

---

## 🧭 Resumen ejecutivo

Sobre **2,000 viajes** con clima y eventos, el resultado es **honesto y aleccionador**: las
features disponibles **no predicen el retraso** (R² ≈ 0 en los tres modelos). El valor del
proyecto está en el **rigor metodológico**: detectar *data leakage*, usar validación cruzada y
diagnosticar correctamente la falta de señal en lugar de reportar un número inflado.

---

## 📊 Detecciones del EDA (Fase 1)

| # | Detección | Evidencia |
|---|---|---|
| 1 | **Retraso medio ~13 min**; la mayoría de viajes registran algún retraso | `01_delay_distribution.png` |
| 2 | **Clima y eventos con impacto modesto** — diferencias pequeñas entre categorías | `02_weather_impact.png`, `03_events_impact.png` |
| 3 | **Correlaciones numéricas ≈ 0** con el retraso (temp, humedad, tráfico…) | `05_correlation.png` |
| 4 | `event_type` con 1,173 nulos = **"sin evento"** → imputado a `None` | `01_EDA` |

---

## 🤖 Detecciones del modelado (Fase 3)

| Modelo | MAE (min) | RMSE | R² |
|---|---|---|---|
| **Linear Regression** ✅ | 7.73 | 9.25 | −0.02 |
| Random Forest | 7.76 | 9.25 | −0.02 |
| XGBoost | 7.88 | 9.41 | −0.05 |

1. **R² ≈ 0 / negativo** → los modelos no superan a predecir la media. Las features no tienen
   poder predictivo sobre el retraso.
2. **Un modelo lineal empata con XGBoost** → no es problema de capacidad del modelo, sino de
   **falta de señal en los datos**.
3. **Validación cruzada 5-fold** confirma que el resultado es estable (no un artefacto del split).

---

## 🎓 Aprendizajes

**Técnicos**
1. **Detectar *data leakage*:** `delayed` correlaciona 0.76 con el target porque **es** el target
   binarizado. Incluirlo habría dado un R² artificialmente alto. Se excluyó.
2. **Validación cruzada en datasets pequeños** (2,000 filas) es imprescindible para no
   sobreinterpretar un único split.
3. **Eliminar columnas de alta cardinalidad** (estaciones, IDs) que no generalizan.
4. **Reutilizar features ya provistas** (`weekday`, `peak_hour`, `season`) en vez de recrearlas.

**De proceso / data literacy**
5. **Un R² bajo bien diagnosticado es un entregable válido.** Saber distinguir "el modelo es malo"
   de "los datos no tienen señal" es una habilidad clave de un analista.
6. **No inflar resultados:** es preferible reportar R² ≈ 0 con honestidad que forzar métricas con
   leakage o overfitting.

---

## ⚠️ Limitaciones y próximos pasos

- El dataset parece **sintético** y de baja señal; conclusiones de negocio limitadas.
- [ ] Conseguir datos reales con mayor señal: **GPS/AVL, ocupación, incidencias, obras**.
- [ ] Reformular como **clasificación** (retraso severo sí/no) si el negocio lo prefiere.
- [ ] Con datos reales: tuning de hiperparámetros y features de interacción clima×hora-pico.

---

*Documento vivo — se actualiza conforme evoluciona el proyecto.*
