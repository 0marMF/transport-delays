# Reporte descriptivo — retrasos de transporte

Retraso medio global: **13.32 min** (desviación 9.29 min).

## Minutos extra por clima (vs. cielo despejado)

| Clima | Retraso medio (min) | Delta vs Clear | n |
|---|---|---|---|
| Rain | 13.62 | +0.12 | 321 |
| Snow | 13.53 | +0.03 | 343 |
| Clear | 13.5 | +0.0 | 343 |
| Storm | 13.29 | -0.21 | 337 |
| Cloudy | 13.07 | -0.43 | 325 |
| Fog | 12.88 | -0.62 | 331 |

## Minutos extra por evento (vs. sin evento)

| Evento | Retraso medio (min) | Delta vs None | n |
|---|---|---|---|
| Protest | 13.9 | +0.37 | 86 |
| Parade | 13.62 | +0.09 | 105 |
| None | 13.53 | +0.0 | 1173 |
| Concert | 12.83 | -0.69 | 203 |
| Sports | 12.82 | -0.71 | 212 |
| Festival | 12.77 | -0.76 | 221 |

## ¿Y como clasificación (retraso severo sí/no)?

| Umbral (min) | % severos | AUC (CV) |
|---|---|---|
| >= 15 | 46.3% | 0.496 |
| >= 18 | 36.4% | 0.505 |
| >= 20 | 29.9% | 0.486 |

## Veredicto

Sin señal utilizable. Las diferencias por clima (<0.62 min) y por evento (<0.76 min) son ruido frente a una desviación de 9.3 min, y reformular como clasificación da AUC ~ 0.5. Clima y eventos de este dataset no explican el retraso.

## Recomendaciones para operadores

- **No usar clima ni eventos de este dataset para anticipar retrasos:** no tienen poder descriptivo (diferencias < 1 min, dentro del ruido) ni predictivo (R² ~ 0, AUC ~ 0.5).
- **Planificar con el promedio + margen:** el retraso medio (~13 min) es estable; operativamente es más sensato asumir ese promedio que confiar en 'predicciones' de estas variables.
- **Para un modelo útil hacen falta datos con señal real:** GPS/AVL en tiempo real, ocupación, incidencias/averías, obras, headway y el retraso de salida observado.
