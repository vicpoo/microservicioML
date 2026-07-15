# Cómo funciona el ML de microservicioMLL

Este documento explica, en orden, todo el flujo del machine learning del proyecto: de qué archivos salen los datos, qué le hacemos, con qué algoritmos entrenamos, y por qué unos modelos son supervisados y otro no. Es el mismo flujo que corre el notebook `notebook_ml_microservicioMLL.ipynb`.

## 1. El flujo completo, de un vistazo

```
Sensores IoT / simulación  ->  Limpieza (nulos)  ->  Features  ->  Entrenamiento  ->  Modelo entrenado  ->  Evaluación  ->  Predicción en producción
   (recolección)                (preprocesamiento)                  (algoritmo)        (artefacto .joblib)   (métricas)      (API /anomalies/detect)
```

Cada flecha de ese diagrama es una sección de este documento.

## 2. De qué archivos salen los datos (y qué genera la limpieza)

Esto es lo que realmente lee y escribe el código en disco, archivo por archivo:

| Archivo | Quién lo escribe | Quién lo lee | Contenido |
|---|---|---|---|
| `data/raw/lecturas_ml_training.csv` | `scripts/generar_dataset.py` | `scripts/train_models.py` | Dataset sintético crudo: 1 fila por lectura simulada, con nulos de sensor incluidos a propósito |
| `data/raw/retroalimentacion_real.csv` | `scripts/exportar_retroalimentacion.py` | `scripts/train_models.py` | Opcional. Lotes reales ya finalizados, reportados por productores (RNF-19). Si no existe, se entrena solo con el sintético |
| `data/processed/lecturas_limpias.csv` | `scripts/train_models.py::cargar_y_limpiar()` | Nadie más lo vuelve a leer; es solo un "dejar constancia" de con qué datos ya limpios se entrenó | Sintético + real combinados, sin nulos, con `delta_temp` ya calculado |
| `app/ml/artifacts/*.joblib` | `scripts/train_models.py` (al final de cada entrenamiento) | `app/services/anomaly_detector.py` y `app/services/predictor.py` | Los 4 modelos ya entrenados, listos para cargar con `joblib.load` |
| `app/ml/artifacts/metricas.json` | `scripts/train_models.py` | Nadie en producción (es para ti, para revisar métricas) | accuracy/f1/rmse/mae de cada modelo |

O sea: la limpieza (`cargar_y_limpiar`) no es un paso invisible — sí escribe un archivo real, `data/processed/lecturas_limpias.csv`, precisamente para que puedas abrirlo y comprobar que ya no tiene nulos.

### 2.1 Cómo limpia los datos, paso a paso

`cargar_y_limpiar()` en `scripts/train_models.py` hace, en este orden:

1. Lee `data/raw/lecturas_ml_training.csv`.
2. Si existe `data/raw/retroalimentacion_real.csv`, lo concatena (`pd.concat`) con el sintético.
3. Descarta filas sin `tipo_proceso` o sin `id_lote` (`dropna(subset=[...])`) — son filas inservibles, no se pueden ni clasificar por proceso ni agrupar por lote.
4. Para cada una de las 6 variables de sensor, rellena los nulos con la **mediana de esa misma columna** (`df[col].fillna(mediana)`). No se usa la media porque un solo valor de sensor disparado (ej. 120°C) movería mucho el promedio; la mediana es más robusta a esos outliers.
5. Calcula `delta_temp = temperatura_grano - temperatura_ambiental`.
6. Guarda el resultado en `data/processed/lecturas_limpias.csv`.

### 2.2 El archivo generado: evidencia real de tu proyecto

Ya tienes un `data/processed/lecturas_limpias.csv` en tu carpeta (1,046,534 bytes), generado por una corrida anterior de tu propio `scripts/train_models.py`. Así se ve realmente:

- **Forma**: 10,654 filas x 15 columnas.
- **Columnas**: `id_lote, tipo_proceso, horas_transcurridas, horas_restantes, temperatura_grano, temperatura_ambiental, humedad_ambiental, humedad_grano, lluvia, luz, _es_anomalia, _severidad, _tipo_anomalia, _calidad_final_lote, delta_temp`
- **Nulos por columna**: 0 en todas — la limpieza sí funcionó, no quedó ningún hueco.

Primeras filas reales de ese archivo:

| id_lote | tipo_proceso | horas_transcurridas | horas_restantes | temperatura_grano | temperatura_ambiental | humedad_ambiental | humedad_grano | lluvia | luz | delta_temp |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | lavado | 194 | 12 | 24.13 | 22.72 | 52.30 | 15.13 | 0.0 | 0.0 | 1.41 |
| 25 | honey | 48 | 376 | 25.67 | 26.83 | 52.83 | 47.96 | 0.0 | 0.0 | -1.16 |
| 24 | lavado | 124 | 53 | 24.89 | 22.11 | 56.41 | 26.54 | 0.0 | 0.0 | 2.78 |

Esto confirma dos cosas: que la limpieza sí corrió de verdad (0 nulos, `delta_temp` ya calculado), y que el archivo que de verdad usa `train_models.py` para entrenar tiene 6 variables de sensor + `tipo_proceso` + `horas_transcurridas`, no solo 2.

### 2.3 Ojo: tu `data/raw/lecturas_ml_training.csv` actual no coincide con esto

Al revisar tu carpeta `data/raw` ahora mismo, el `lecturas_ml_training.csv` que tienes ahí **no** tiene esas 6 variables de sensor: tiene solo `id_lectura, id_sensor, id_lote, temperatura, humedad, timestamp, _tipo_proceso, _es_anomalia, _tipo_anomalia` (9 columnas, 4,608 filas). Esa es la salida de tu notebook `01_generacion_dataset.ipynb` (el prototipo "Paso 1", con solo temperatura/humedad), no la del `scripts/generar_dataset.py` final de 6 variables.

En algún momento generaste el dataset correcto (por eso existe `data/processed/lecturas_limpias.csv` con las 6 variables), pero después volviste a correr el notebook prototipo y sobrescribió `data/raw/lecturas_ml_training.csv` con la versión vieja de 2 variables. Si ahora mismo corrieras `scripts/train_models.py` de tu proyecto real, fallaría al buscar la columna `temperatura_grano` (no existe en el CSV actual). Para arreglarlo: vuelve a correr `scripts/generar_dataset.py` (el de 6 variables, no el notebook viejo) antes de reentrenar.

## 3. Qué hacemos en cada paso

### 3.1 Recolección de datos

El kit de sensores IoT del secado de café mide 6 variables cada cierto tiempo:

- `temperatura_grano` (sensor DS18B20)
- `temperatura_ambiental` y `humedad_ambiental` (sensor BME280)
- `humedad_grano` (sensor capacitivo)
- `lluvia` (sensor FC-37, normalizado de 0 a 1)
- `luz` (sensor BH1750, en lux)

En producción estas lecturas se guardan en la tabla `lecturas_ambientales` de la base de datos. Para poder entrenar sin depender de meses de datos reales, `scripts/generar_dataset.py` **simula** lotes completos de secado (lavado, honey, natural) con esas mismas 6 variables, incluyendo fallos de sensor y valores nulos ocasionales, tal como pasaría con hardware real. El resultado se guarda en `data/raw/lecturas_ml_training.csv` (ver sección 2).

### 3.2 Preprocesamiento

Antes de que cualquier modelo vea los datos (detalle completo en la sección 2.1):

- Se imputan los nulos de sensores con la **mediana** de cada columna.
- Se calcula una variable derivada: `delta_temp = temperatura_grano - temperatura_ambiental`.
- La variable categórica `tipo_proceso` (lavado/honey/natural) se convierte a números con **one-hot encoding**.

### 3.3 Features (las variables que ve el modelo)

| Feature | Tipo | Por qué se incluye |
|---|---|---|
| `temperatura_grano`, `temperatura_ambiental`, `humedad_ambiental`, `humedad_grano`, `lluvia`, `luz` | Numéricas | Las 6 lecturas crudas de los sensores |
| `delta_temp` | Numérica derivada | Detecta fluctuaciones térmicas bruscas |
| `tipo_proceso` | Categórica | El rango "ideal" de temperatura/humedad cambia según el proceso |
| `horas_transcurridas` | Numérica | Solo la usan los modelos de tiempo restante y calidad final (importa en qué punto del secado va el lote) |

### 3.4 Algoritmo y entrenamiento

Aquí es donde entran los modelos. Se explican a detalle en la sección 4.

### 3.5 Modelo entrenado

Cada modelo, ya entrenado, se guarda como un archivo `.joblib` en `app/ml/artifacts/` (ver tabla de la sección 2). Guardar el modelo entrenado es lo que permite que el servicio en producción no tenga que reentrenar en cada petición: simplemente carga el archivo y llama `.predict()`.

### 3.6 Evaluación

Antes de dar por bueno un modelo, se mide qué tan bien predice sobre datos que **no vio** durante el entrenamiento (el "holdout"). Aquí no partimos filas al azar: usamos `GroupShuffleSplit` agrupado por `id_lote`, para que ningún lote quede repartido entre entrenamiento y prueba (si no, el modelo "haría trampa" viendo lecturas de un lote que ya conoce de otras horas del mismo lote).

### 3.7 Predicción (uso en producción)

Cuando llega una lectura nueva, el servicio corre el modelo ya entrenado sobre esa lectura y devuelve: si es una anomalía, qué tan grave, cuánto tiempo falta de secado, y qué calidad final se espera. Esto es lo que expone el endpoint `POST /anomalies/detect`.

## 4. Los 4 entrenamientos, uno por uno (y por qué el notebook los separa en dos partes)

El notebook (`notebook_ml_microservicioMLL.ipynb`) ya **no mezcla** los 4 modelos en una sola sección: están divididos en **Parte A (no supervisado)** y **Parte B (supervisado)**, cada una con su propio ciclo completo de algoritmo → entrenamiento → evaluación → predicción, para que sea evidente que son dos paradigmas de aprendizaje distintos y no un solo bloque homogéneo.

| # | Parte del notebook | Modelo | Algoritmo | Qué predice | Etiqueta que usa (`y`) |
|---|---|---|---|---|---|
| 1 | **Parte A** | Detector de outliers | **IsolationForest** | Si una lectura es "rara" comparada con el resto | Ninguna (no supervisado) |
| 2 | **Parte B** | Clasificador de tipo de anomalía | **RandomForestClassifier** | Qué tipo de problema hay (temperatura alta, lluvia, moho, etc.) o "normal" | `_tipo_anomalia` |
| 3 | **Parte B** | Regresor de tiempo restante | **RandomForestRegressor** | Cuántas horas faltan para terminar el secado | `horas_restantes` |
| 4 | **Parte B** | Clasificador de calidad final | **RandomForestClassifier** | Calidad final esperada del lote (excelente/buena/regular/baja) | `_calidad_final_lote` |

Las etiquetas de los modelos 2 y 4 no las puso una persona a mano: salen de un **motor de reglas de dominio** (umbrales de temperatura, humedad, lluvia, etc. tomados del documento de calidad del café) que se le aplica a cada lectura simulada. Es decir, el Random Forest aprende a imitar (y generalizar más allá de) esas reglas.

La Parte 0 del notebook (recolección, preprocesamiento, validación de datos) es compartida por ambas partes — no se duplica, porque tanto el modelo no supervisado como los 3 supervisados parten del mismo dataset limpio.

## 5. ¿Qué es supervisado y qué no, y por qué

**Supervisado** quiere decir que, durante el entrenamiento, el modelo ve pares de (features, respuesta correcta) y ajusta sus parámetros para acercarse a esa respuesta. Sirve para **clasificar** (elegir una categoría) o **predecir** un valor numérico (regresión).

**No supervisado** quiere decir que el modelo **nunca ve la respuesta correcta**: solo mira cómo se distribuyen los datos y encuentra patrones o puntos raros por su cuenta.

| Modelo | ¿Supervisado? | Evidencia en el código |
|---|---|---|
| RandomForestClassifier (tipo de anomalía) | **Sí** | `pipe.fit(X, y)` donde `y = df["_tipo_anomalia"]` — el modelo recibe la respuesta correcta de cada fila |
| RandomForestRegressor (tiempo restante) | **Sí** | `pipe.fit(X, y)` donde `y = df["horas_restantes"]` — un número ya conocido |
| RandomForestClassifier (calidad final) | **Sí** | `pipe.fit(X, y)` donde `y = df["_calidad_final_lote"]` — una categoría ya conocida |
| IsolationForest | **No** | `modelo.fit(X)` — nótese que **no hay una `y`**; nunca se le dice cuáles filas eran anomalías. La proporción de anomalías del dataset solo se usa para calibrar un parámetro (`contamination`), no para entrenar |

En producción, estos modelos no trabajan solos: se combinan con un **motor de reglas** (determinista, no es machine learning) en un ensamble de 3 piezas:

1. **Reglas de dominio** — umbrales fijos y conocidos (ej. "si hay lluvia, es crítico"). Siempre ganan cuando detectan algo.
2. **RandomForest (supervisado)** — generaliza patrones aprendidos de miles de ejemplos etiquetados, para casos que las reglas no cubren exactamente.
3. **IsolationForest (no supervisado)** — atrapa lecturas raras que ni las reglas ni el clasificador conocían de antemano.

### Nota sobre K-Means, DBSCAN, Jerárquico, PCA

Estos 4 son algoritmos no supervisados de **clustering** (agrupar puntos parecidos) o reducción de dimensionalidad. El proyecto no usa ninguno de ellos: usa **IsolationForest**, que también es no supervisado pero resuelve un problema distinto — detección de anomalías/outliers, no agrupamiento.

## 6. Qué usamos (resumen técnico)

- **Lenguaje y librerías**: Python, `pandas`/`numpy` (datos), `scikit-learn` (modelos), `matplotlib`/`seaborn` (gráficas), `joblib` (guardar modelos entrenados), `fastapi`/`httpx` (probar el endpoint real en el notebook).
- **Algoritmos**: `RandomForestClassifier` x2, `RandomForestRegressor` x1, `IsolationForest` x1.
- **Validación**: `GroupShuffleSplit` agrupado por lote (evita fuga de datos entre entrenamiento y prueba).
- **Métricas**: `accuracy`, `f1_macro`, `classification_report` y matriz de confusión para los clasificadores; `RMSE`/`MAE` y gráfico predicho-vs-real para el regresor; tasa de outliers detectados para el IsolationForest.
- **Servicio**: FastAPI, expone `POST /anomalies/detect` (manual/pruebas) y `POST /internal/lecturas/nuevas` (disparado por el Servicio Gestor en producción).

## 7. Cómo comprobamos que los datos estén bien

Esto fue una pregunta directa del profesor: no basta con generar datos, hay que probar activamente que están correctos antes de entrenar con ellos. Se agregó una celda de validación al notebook (Parte 0, sección "0.7 Como comprobamos que los datos estan bien") con 6 comprobaciones reales, cada una con `assert` — si algo falla, el notebook se detiene ahí en vez de entrenar modelos sobre datos malos:

1. **Estructura completa**: las 15 columnas esperadas existen en `df_limpio` (ninguna falta, ninguna sobra sin explicación).
2. **Cero nulos** en las columnas numéricas usadas como features, después de la limpieza.
3. **Rangos físicamente válidos**: `humedad_ambiental` y `humedad_grano` entre 0-100%, `lluvia` entre 0-1, `luz` no negativa. Aquí hubo un matiz importante: `generar_dataset.py` inyecta a propósito fallos de sensor (~0.4% de probabilidad) que ponen valores imposibles (ej. humedad negativa) para poder entrenar al modelo a reconocerlos. Por eso la validación no exige "cero valores fuera de rango" — exige que sean **menos del 1%** del total **y** que el motor de reglas los haya marcado correctamente como `crítico`. Es una prueba más fuerte: no solo mide que los datos estén limpios, sino que el sistema de detección de anomalías realmente funciona sobre esos casos.
4. **Sin filas duplicadas** por `(id_lote, horas_transcurridas)` — cada lectura de sensor debe ser única en el tiempo dentro de su lote.
5. **Balance de clases razonable**: la proporción de anomalías debe estar entre 5% y 20% (el diseño apunta a ~90/10), para asegurar que el dataset sea realista y entrenable.
6. **Categorías válidas**: `tipo_proceso` solo contiene `lavado`, `honey` o `natural`.

Al ejecutar el notebook, las 6 pasan con salida explícita, por ejemplo:
```
[OK] 'humedad_ambiental': 10 filas fuera de rango (0.09%), y las 10 quedaron correctamente marcadas como 'critico' por las reglas.
Todas las validaciones de calidad de datos pasaron. Se puede entrenar con confianza.
```

## 8. Cómo se entrenan los modelos (y por qué ahora es rápido)

El profesor notó que el notebook tardaba demasiado (llegó a 40 minutos) y pidió usar concurrencia. Había dos partes lentas: generar los datos sintéticos lote por lote, y entrenar los modelos uno tras otro. Se paralelizaron ambas, con cuidado de no repetir el problema original. Hay **dos bloques de concurrencia independientes** en el notebook:

**1) Generación de datos (concurrente con hilos, Parte 0):**
- Antes: un ciclo `for` secuencial creaba cada lote uno por uno usando un único generador de números aleatorios (`RNG`) global.
- Ahora: se usa `concurrent.futures.ThreadPoolExecutor(max_workers=8)` para simular hasta 8 lotes al mismo tiempo. El problema de hacer esto con un RNG compartido es que no es seguro entre hilos (dos lotes podrían "pisarse" y dejar de ser reproducibles). Se resolvió con `numpy.random.SeedSequence(42).spawn(n)`, que genera N flujos de números aleatorios independientes y reproducibles — cada lote recibe su propio generador (`rng`) sin compartir estado con los demás.
- Resultado real en el notebook: 72 lotes simulados en paralelo en **0.3 segundos**.

**2) Entrenamiento de los 3 modelos supervisados (concurrente con hilos, Parte B):**
- El `IsolationForest` de la Parte A se entrena solo (es un único modelo, no hay nada que paralelizar ahí). La concurrencia de entrenamiento aplica a los **3 `RandomForest*` de la Parte B**, que sí son independientes entre sí.
- Antes: los 3 usaban `n_jobs=-1` (usar todos los núcleos) y se entrenaban uno detrás del otro. En Windows, `n_jobs=-1` dispara el backend `loky` de `joblib`, que crea procesos nuevos del sistema operativo — con overhead altísimo en Windows/Jupyter, y fue la causa real de los 40 minutos.
- Ahora: cada modelo se entrena con `n_jobs=1` (un solo hilo interno, sin overhead de crear procesos), pero los 3 entrenamientos (clasificador de tipo de anomalía, regresor de tiempo restante, clasificador de calidad final) se lanzan **a la vez** en 3 hilos con `ThreadPoolExecutor(max_workers=3)`. Esto funciona porque `scikit-learn`/`numpy` liberan el GIL de Python durante los cálculos pesados en C/Cython, así que los hilos sí logran ejecutarse en paralelo real para este tipo de carga — sin el costo de crear procesos nuevos.
- Resultado real en el notebook: IsolationForest (Parte A) entrena solo en **0.3 segundos**; los 3 RandomForest (Parte B) entrenan en paralelo en **16.2 segundos**; el notebook completo (44 celdas, incluyendo la prueba de la API real) corre en **~21 segundos**, contra los 40 minutos originales.

## 9. Qué devuelve la API (y ahora se prueba con un endpoint real, no solo con un JSON de ejemplo)

Antes, esta sección solo mostraba un diccionario armado a mano con valores de ejemplo. Eso ya no es así: en la **Parte C** del notebook se construye una app **FastAPI real** (mismo endpoint, mismos schemas Pydantic que el proyecto) y se le hacen peticiones HTTP de verdad con `fastapi.testclient.TestClient` — la forma estándar de probar endpoints de FastAPI sin levantar un servidor aparte ni depender de la base de datos. El notebook hace `client.get("/health")` y `client.post("/api/v1/anomalies/detect", json=...)` para los 4 casos de prueba, e imprime el `status_code` y el JSON real que devolvió el endpoint.

El endpoint `POST /api/v1/anomalies/detect` responde con el esquema `InferenceResponse` (`app/schemas/inference_response.py`). Esta es la respuesta **real** que devolvió el endpoint de prueba (no un ejemplo escrito a mano) para el caso "Crítico (lluvia detectada)":

```json
{
  "id_inferencia": 3,
  "id_lote": null,
  "es_anomalia": true,
  "nivel_severidad": "critico",
  "score_isolation_forest": -0.1294,
  "confianza_ml": 91.3,
  "variables_contribuyentes": ["humedad_ambiental", "lluvia"],
  "mensaje": "Severidad detectada: critico.",
  "recomendaciones": [
    {"tipo": "humedad_ambiental_alta", "texto": "Voltea el cafe con mayor frecuencia..."},
    {"tipo": "lluvia_detectada", "texto": "Prioridad maxima: cubre el lote con plastico..."}
  ],
  "prediccion": {"tiempo_estimado_horas": 92.4, "calidad_estimada": "baja", "confianza": 41.2},
  "alerta_generada": true,
  "id_alerta": 101,
  "notificacion_email_enviada": false,
  "modelo_version": "2.0.0",
  "fecha_inferencia": "2026-07-15T20:42:19.259Z"
}
```

Campos clave: `es_anomalia` (bool, resumen rápido de si hay algo anómalo), `nivel_severidad` (`normal` / `advertencia` / `riesgo` / `critico`, viene de reglas + ML combinados), `score_isolation_forest` (salida del modelo no supervisado de la Parte A: más negativo = más atípico), `confianza_ml` (probabilidad del RandomForest de tipo de anomalía, Parte B), `variables_contribuyentes` (qué variables dispararon las reglas), `recomendaciones` (texto generado por `recommender.py`), `prediccion` (tiempo restante y calidad, de los otros 2 RandomForest de la Parte B), y `alerta_generada`/`id_alerta` (si se creó una alerta — solo ocurre en niveles riesgo/crítico).

## 9.1 Evaluación reforzada de los modelos supervisados

Para que la evaluación no se quede solo en accuracy/F1/RMSE, la Parte B del notebook agrega:

- **Reporte de clasificación** (`classification_report`) por clase para los dos clasificadores (tipo de anomalía y calidad final), no solo un promedio global — así se ve si el modelo falla más en alguna categoría específica (por ejemplo, clases minoritarias).
- **Matriz de confusión** (heatmap) para ambos clasificadores, mostrando exactamente qué categorías se confunden entre sí.
- **Gráfico predicho vs. real** para el regresor de tiempo restante, para ver visualmente qué tan cerca están las predicciones de la línea ideal.
- **Importancia de features** (`feature_importances_`) para los 3 modelos, mostrando qué variables pesan más en cada predicción — útil para justificar, frente a un revisor técnico, por qué el modelo decide lo que decide.

## 9.2 Corrección: error "Input contains NaN" al entrenar (Parte B)

Al correr el notebook en otra máquina apareció `ValueError: Input contains NaN` dentro de uno de los 3 entrenamientos concurrentes de la Parte B. La causa más probable era una combinación frágil en `rebalancear_90_10`: usaba `groupby(...).apply(lambda ...)`, cuyo comportamiento de reindexado puede variar entre versiones de pandas y, en el peor caso, introducir una fila mal alineada. Se corrigió así:

1. **`rebalancear_90_10` ya no usa `groupby().apply()`**: se reemplazó por un bucle explícito sobre cada grupo de `_tipo_anomalia` + `pd.concat`, que se comporta igual sin importar la versión de pandas instalada.
2. **`cargar_y_limpiar` ahora tiene una salvaguarda final**: después de imputar las 6 columnas de sensores con la mediana, elimina (con aviso impreso) cualquier fila que aún conserve un NaN en alguna columna crítica (features o etiquetas). En condiciones normales esto elimina 0 filas — es una red de seguridad, no un parche que oculte el problema.
3. **Cada una de las 3 funciones de entrenamiento de la Parte B ahora valida explícitamente** (`_verificar_sin_nans`) que no haya NaN en `X`/`y` justo antes de `pipe.fit()`. Si algo se cuela, el error ahora dice exactamente qué modelo, cuántas filas y qué columnas — en vez del `ValueError` genérico de scikit-learn señalando una línea interna de `sklearn/utils/validation.py`.

Con esta corrección se volvió a ejecutar el notebook completo de punta a punta (44 celdas, 0 errores) para confirmar que el fix no rompió nada más.

## 10. Cómo probarlo con Postman o Insomnia

Se generó el archivo `microservicioMLL.postman_collection.json` (formato Postman Collection v2.1, compatible con importar directo en Insomnia también). Incluye:

- `GET /health` — verifica que el servicio esté vivo.
- `POST /api/v1/anomalies/detect` — 5 variantes: caso normal, advertencia (temperatura alta), crítico por lluvia, crítico por valor de sensor imposible, y un caso negativo esperando 403 (usuario que no es dueño del lote).
- `POST /api/v1/internal/lecturas/nuevas` — simula el aviso que envía el Servicio Gestor cuando llega una lectura nueva.
- `POST /api/v1/internal/lotes/{id_lote}/resultado-real` — registra la retroalimentación real de calidad/tiempo cuando termina un secado (RNF-19).
- `GET /api/v1/anomalies`, `GET /api/v1/anomalies/{id_lote}/predicciones`, `GET /api/v1/anomalies/{id_lote}/recomendaciones` — endpoints de historial.

Pasos para usarlo:
1. Levantar el servicio localmente (`uvicorn app.main:app --reload` o el comando que use el proyecto).
2. Importar `microservicioMLL.postman_collection.json` en Postman (Import → File) o en Insomnia (Import from File — Insomnia lee colecciones Postman v2.1 sin problema).
3. Ajustar las variables de la colección si hace falta: `base_url` (por defecto `http://127.0.0.1:8000`), `internal_api_key` (debe coincidir con `INTERNAL_API_KEY` del `.env` si el servicio la exige) y `id_lote` (un lote real que exista en la base de datos, para los requests que dependen de un lote existente).
4. Correr las peticiones en orden: primero `health`, luego los casos de `/detect` (no requieren lote existente salvo el de lluvia), y por último los de historial/internos si ya hay datos guardados.

Nota: la colección de Postman prueba el **servicio real completo** (con base de datos). El notebook, en cambio, prueba una **réplica local del endpoint** con `TestClient` (sin base de datos), pensada para verificar rápidamente que el ensamble de ML responde con el esquema correcto.
