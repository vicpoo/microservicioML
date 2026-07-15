# Verificación del pipeline de Machine Learning — microservicioMLL

## Veredicto rápido

Sí cumple. El microservicio implementa las 8 partes clásicas del pipeline de ML (recolección → preprocesamiento → features → algoritmo → entrenamiento → modelo entrenado → evaluación → predicción), combina **modelos supervisados** (clasificación y regresión) con **un modelo no supervisado** (detección de outliers), y además suma un motor de reglas de dominio que actúa como respaldo determinista del ML. El único matiz: los notebooks subidos (`01_generacion_dataset*.ipynb`, `01_evidencia_microservicio.ipynb`) son un prototipo temprano ("Paso 1", solo 2 variables y un Isolation Forest) que ya quedó superado por el pipeline final de 6 variables y 4 modelos que sí está en `scripts/` y `app/services/`. Se explica al final en "Notas y observaciones".

---

## 1. Recolección de datos

| Dónde | Qué hace |
|---|---|
| `scripts/generar_dataset.py` → `simular_lote()` | Genera datos **sintéticos** simulando lotes completos de secado (lavado/honey/natural), con 5 sensores físicos del kit IoT: BME280 (temperatura/humedad ambiental), DS18B20 (temperatura de grano), BH1750 (luz), FC-37 (lluvia) y sensor capacitivo (humedad de grano). Incluye fallos de sensor y nulos simulados. Salida: `data/raw/lecturas_ml_training.csv`. |
| `app/api/routes/internal.py` → `registrar_resultado_real()` (`POST /internal/lotes/{id_lote}/resultado-real`) | Recolección de **datos reales** de campo: el productor reporta calidad final y tiempo real de secado de un lote ya terminado (RNF-19). Se guarda en la tabla `retroalimentacion_ml`. |
| `scripts/exportar_retroalimentacion.py` | Convierte esos datos reales de `retroalimentacion_ml` al mismo esquema de columnas del dataset sintético, para poder combinarlos al reentrenar. |

Es decir: hay dos fuentes de datos, una sintética (para arrancar el modelo sin datos reales) y una real (para ir sustituyéndola con retroalimentación de productores).

## 2. Preprocesamiento de datos

| Dónde | Qué hace |
|---|---|
| `scripts/train_models.py` → `cargar_y_limpiar()` | Combina el CSV sintético con el CSV de retroalimentación real, elimina filas sin `tipo_proceso`/`id_lote`, **imputa nulos con la mediana** en las 6 variables numéricas, calcula la columna derivada `delta_temp`, y guarda el resultado limpio en `data/processed/lecturas_limpias.csv`. |
| `scripts/train_models.py` → `_preprocesador()` | `ColumnTransformer` con `OneHotEncoder` para la variable categórica `tipo_proceso`, empaquetado dentro de cada `Pipeline` de sklearn (así el preprocesamiento viaja junto con el modelo entrenado). |
| `app/services/preprocessor.py` → `Preprocessor.transform()` | Preprocesamiento **en producción**: arma el vector de 6 variables a partir del payload recibido (rellena con `0.0` si falta algo) y calcula `delta_temp`, para que el vector que llega al modelo tenga siempre la misma forma que en entrenamiento. |

## 3. Features (variables)

- **Numéricas** (`NUMERIC_FEATURES`): `temperatura_grano`, `temperatura_ambiental`, `humedad_ambiental`, `humedad_grano`, `lluvia`, `luz`, y la derivada `delta_temp` (= temperatura_grano − temperatura_ambiental).
- **Categórica**: `tipo_proceso` (lavado / honey / natural), codificada con one-hot.
- **Features temporales/contextuales adicionales**, calculadas solo al momento de inferir (`app/api/routes/inference.py` → `_contexto_historico()`): `delta_temp_reciente` (variación respecto a la lectura anterior) y `delta_humedad_grano_24h` (cuánto bajó la humedad de grano en 24h), usadas por el motor de reglas para detectar fluctuación térmica y secado estancado.
- **`horas_transcurridas`**: feature adicional para los modelos de tiempo restante y calidad final.

## 4. Algoritmo (con qué aprende el modelo)

| Modelo | Algoritmo | Tipo | Para qué |
|---|---|---|---|
| `isolation_forest.joblib` | **IsolationForest** | No supervisado | Detecta lecturas atípicas/outliers que ni las reglas ni el clasificador conocen |
| `rf_tipo_anomalia.joblib` | **RandomForestClassifier** | Supervisado | Clasifica el tipo de anomalía (incluye "normal") |
| `rf_tiempo_restante.joblib` | **RandomForestRegressor** | Supervisado | Predice horas restantes de secado |
| `rf_calidad.joblib` | **RandomForestClassifier** | Supervisado | Predice la calidad final estimada del lote |
| `app/services/rules.py` | Motor de reglas (umbrales fijos, no es ML) | — | Fuente autoritativa para casos conocidos (Cuadro 9 del documento de dominio); el ML complementa/generaliza |

`app/services/anomaly_detector.py` combina los tres primeros en un **ensamble**: reglas + RandomForest + IsolationForest, donde las reglas mandan cuando detectan algo, y el ML puede *elevar* la severidad cuando reconoce un patrón que las reglas no cubren.

## 5. Entrenamiento

Todo ocurre en `scripts/train_models.py`:

- `entrenar_isolation_forest(df)`: entrena sobre **todas** las filas (no supervisado, no usa la etiqueta para ajustar el modelo); usa la tasa de anomalías del dataset (`df['_es_anomalia'].mean()`) solo para calibrar el hiperparámetro `contamination`.
- `entrenar_clasificador_tipo(df)`, `entrenar_regresor_tiempo(df)`, `entrenar_clasificador_calidad(df)`: para los 3 modelos supervisados, usa `GroupShuffleSplit` agrupando por `id_lote` (25% test), para que **ningún lote se filtre entre train y test** (evita fuga de datos). Después de medir métricas en el holdout, cada pipeline se **reentrena con el 100% de los datos** para el artefacto final de producción.
- Cada modelo es un `Pipeline` de sklearn completo (preprocesamiento + estimador), así el servicio solo arma un DataFrame de una fila y llama `.predict()`.

## 6. Modelo entrenado

- Los 4 artefactos se serializan con `joblib.dump()` en `app/ml/artifacts/` (`isolation_forest.joblib`, `rf_tipo_anomalia.joblib`, `rf_tiempo_restante.joblib`, `rf_calidad.joblib`).
- Se cargan en producción con `joblib.load()` desde `app/services/anomaly_detector.py` (`AnomalyDetector.__init__`) y `app/services/predictor.py` (`Predictor.__init__`).
- Hay versión de modelo (`modelo_version` en `app/core/config.py`) y un registro en la tabla `modelos_ml` (`app/services/notifier.py` → `get_or_create_modelo()`), para trazabilidad de qué versión generó cada predicción.

## 7. Evaluación

- `scripts/train_models.py`: para los clasificadores, `accuracy_score` y `f1_score(average="macro")`; para el regresor de tiempo, `RMSE` y `MAE`. Todo calculado sobre el holdout **agrupado por lote** (no por fila suelta), y persistido en `app/ml/artifacts/metricas.json`.
- Para IsolationForest (no supervisado, sin etiqueta real de "outlier verdadero") se reporta una métrica proxy: `tasa_outliers_detectados`.
- `scripts/check_dataset.py`: chequeo de calidad del dataset (nulos, balance de clases, distribución de severidad) — es control de calidad de datos, no evaluación de modelo, pero es parte del mismo control de calidad del pipeline.
- `tests/test_api.py`: pruebas de integración que validan que el pipeline completo (reglas + ML) clasifique correctamente severidades esperadas (`critico`, `advertencia`) en escenarios conocidos — funciona como una evaluación funcional end-to-end además de las métricas offline.

## 8. Predicción (inferencia en producción)

- `app/services/anomaly_detector.py` → `AnomalyDetector.predict()`: corre reglas + `predict_proba` del RandomForest + `predict`/`decision_function` del IsolationForest, y combina todo en severidad final + variables contribuyentes.
- `app/services/predictor.py` → `Predictor.predecir()`: `predict()` del regresor de tiempo y `predict_proba()` del clasificador de calidad.
- `app/services/recommender.py`: traduce las alertas detectadas a texto accionable (basado en reglas, no en ML).
- Todo se orquesta en `app/api/routes/inference.py` → `ejecutar_pipeline()`, invocado por dos endpoints:
  - `POST /api/v1/anomalies/detect` — prueba manual (curl/Postman/tests).
  - `POST /api/v1/internal/lecturas/nuevas` — disparado por el Servicio Gestor cuando llega una lectura real.
- El resultado de la predicción se persiste (`predicciones`, `alertas`, `recomendaciones`, `inferencias_ml`) y opcionalmente dispara un correo (`notifier.enviar_email_alerta`).

---

## Supervisado vs. no supervisado — checklist contra tu lista de algoritmos

**Supervisado** = aprende comparando entradas con una respuesta ya conocida (clasificar o predecir un valor). **No supervisado** = busca patrones o grupos en datos sin etiqueta.

| Algoritmo de tu lista | ¿Se usa en el proyecto? | Dónde |
|---|---|---|
| Random Forest | ✅ Sí (3 veces) | `rf_tipo_anomalia`, `rf_tiempo_restante`, `rf_calidad` |
| Árbol de Decisión | ❌ No directo (Random Forest es un ensamble de árboles, pero no se usa un árbol suelto) | — |
| Regresión Lineal | ❌ No | — |
| Regresión Logística | ❌ No | — |
| SVM | ❌ No | — |
| KNN | ❌ No | — |
| Naive Bayes | ❌ No | — |
| Redes Neuronales | ❌ No | — |
| K-Means | ❌ No | — |
| DBSCAN | ❌ No | — |
| Clustering Jerárquico | ❌ No | — |
| PCA | ❌ No | — |

El proyecto **no usa ninguno de los 4 algoritmos no supervisados de tu lista** (K-Means, DBSCAN, Jerárquico, PCA — todos de *clustering* o reducción de dimensionalidad). En su lugar usa **IsolationForest**, que también es no supervisado (no usa etiquetas para ajustar el modelo) pero resuelve un problema distinto: *detección de anomalías/outliers* en vez de *agrupamiento*. Si tu rúbrica exige explícitamente uno de esos 4 algoritmos, este proyecto no lo cubre tal cual — habría que agregar, por ejemplo, un K-Means o PCA como paso adicional de análisis exploratorio, aunque no sea el que corre en producción.

Resumiendo el balance real del proyecto:

- **3 modelos supervisados** (Random Forest, en sus 2 sabores: `RandomForestClassifier` ×2 y `RandomForestRegressor` ×1) — todos aprenden comparando features contra una etiqueta ya conocida (`_tipo_anomalia`, `_calidad_final_lote`, `horas_restantes`).
- **1 modelo no supervisado** (`IsolationForest`) — no usa etiqueta para entrenar, solo busca qué lecturas se separan del resto.

## Notas y observaciones

1. **Las etiquetas no son humanas, son generadas por reglas.** `_es_anomalia`, `_severidad` y `_tipo_anomalia` del dataset sintético se calculan aplicando el motor de reglas (`app/services/rules.py`) sobre la simulación física, no por etiquetado manual. Esto es una decisión de diseño razonable (permite arrancar sin datos reales), pero significa que el Random Forest, en el fondo, está aprendiendo a **imitar las reglas** y a generalizar un poco más allá de ellas — vale la pena mencionarlo si te preguntan de dónde salen las etiquetas.
2. **Los notebooks subidos son un prototipo anterior, no el pipeline final.** `01_generacion_dataset.ipynb` / `..._executed.ipynb` y `01_evidencia_microservicio.ipynb` trabajan solo con 2 variables (`temperatura`, `humedad`), un único `IsolationForest` y todavía incluyen el campo `viento` (que el código final eliminó explícitamente porque el kit de sensores real no tiene anemómetro). El pipeline que sí cumple con las 8 partes completas y el ensamble de 4 modelos es el que vive en `scripts/generar_dataset.py`, `scripts/train_models.py` y `app/services/`. Si vas a entregar evidencia, conviene aclarar que los notebooks documentan el "Paso 1" exploratorio y no reflejan ya el estado actual del microservicio.
3. **RandomForest no es solo "Random Forest".** Aparece en 3 roles distintos (clasificación de tipo de anomalía, clasificación de calidad, regresión de tiempo restante), lo cual es una buena forma de mostrar que entiendes que el mismo algoritmo puede resolver clasificación y regresión, solo cambiando la variable objetivo.
4. **La evaluación evita fuga de datos.** El uso de `GroupShuffleSplit` por `id_lote` (en vez de partir filas al azar) es importante: como cada lote genera muchas filas (una por cada paso de 2h), partir por fila filtraría información del mismo lote entre train y test e infla artificialmente las métricas.
