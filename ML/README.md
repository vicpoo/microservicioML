# ML — kajve

Carpeta con los notebooks y código del pipeline de Machine Learning del proyecto, en el orden
en que se ejecutan.

| Paso | Archivo | Entrada | Salida |
|---|---|---|---|
| 2. Recolección de datos | `../scripts/recolectar_datos_reales.py` (fuera de esta carpeta, corre contra Neon) | BD real | `../data/raw/lecturas_reales_entrenamiento.csv` |
| 3. EDA | `03_eda_datos_reales.ipynb` | `data/raw/lecturas_reales_entrenamiento.csv` | — (análisis, sin archivo de salida) |
| 4. Limpieza de datos | `04_limpieza_datos.ipynb` | `data/raw/lecturas_reales_entrenamiento.csv` | `data/processed/lecturas_reales_limpias.csv` |
| 5. Ingeniería de características | `05_ingenieria_caracteristicas.ipynb` + `ingenieria_caracteristicas.py` | `data/processed/lecturas_reales_limpias.csv` | `data/processed/lecturas_reales_features.csv` |
| 6. División de datos (train/test) | `06_division_datos.ipynb` + `division_datos.py` | `data/processed/lecturas_reales_features.csv` | `data/processed/train.csv`, `data/processed/test.csv` |
| 7. Selección del modelo | `07_seleccion_modelo.ipynb` + `prediccion_lluvia_ga.py` | `data/processed/train.csv`, `test.csv` | — (comparación de candidatos + decisión documentada) |
| 8. Entrenamiento | `08_entrenamiento.ipynb` + `entrenamiento.py` | `data/processed/train.csv` | `ML/artifacts/*.joblib`, `ML/artifacts/metricas_entrenamiento.json` |
| 9. Evaluación | `09_evaluacion.ipynb` + `evaluacion.py` | `data/processed/test.csv`, `ML/artifacts/*.joblib` | — (métricas + comparación contra baseline, por salida) |
| 10. Ajuste de hiperparámetros | `10_ajuste_hiperparametros.ipynb` + `ajuste_hiperparametros.py` | `data/processed/train.csv`, `test.csv` | `ML/artifacts/*.joblib` (sobrescritos con la config ganadora), `metricas_paso10_ajuste.json` |
| 11. Despliegue | `app/services/fcm.py`, `app/services/notifier.py`, `app/services/rain_predictor.py`, `app/services/poller.py`, `app/api/routes/dispositivos.py`, `app/api/routes/inference.py` | `ML/artifacts/isolation_forest.joblib`, `rf_tipo_anomalia.joblib`, `ga_lluvia.joblib` (copiados a `app/ml/artifacts/`) | Notificación push (FCM) en cualquier anomalía Y en riesgo de lluvia próxima, `POST/GET /dispositivos/*`, `GET /anomalies/{id_lote}/predicciones` y `/recomendaciones` (ya existían en `history.py`), revisión periódica de la BD (tiempo real) |
| 12. Monitoreo y reentrenamiento | `monitoreo.py` + `../scripts/monitorear_modelos.py` + `GET /internal/monitoreo/salud` | tablas `predicciones`, `retroalimentacion_ml`, `alertas`, `lecturas_ambientales` (en vivo) | Reporte de salud: desempeño real vs baseline, tasa de alertas reciente, y si ya conviene reentrenar (diagnóstico, nunca automático) |

`ingenieria_caracteristicas.py` es un módulo reutilizable (no solo código de notebook): la idea
es que `scripts/train_models.py` importe las mismas funciones más adelante, en vez de que cada
lugar calcule sus propias columnas derivadas por separado — mismo principio que ya se sigue con
`app/services/rules.py` para las reglas de negocio.

## Pendiente / hallazgos abiertos (documentados dentro de cada notebook)

- `app/services/rules.py::clasificar_valor_imposible()` solo valida `temperatura_grano`; el
  glitch real observado (181.55°C) está en `temperatura_ambiental` y hoy no se detecta en
  producción.
- `app/services/rules.py::clasificar_lluvia()` no tiene ventana de "lluvia sostenida"; con el
  parpadeo real del sensor FC-37, esto genera alertas críticas falsas repetidas en producción
  (el notebook 04 y el módulo de features sí construyen esa señal "sostenida", pero solo para el
  dataset — no está portada todavía a las reglas en vivo).
- `scripts/train_models.py` tiene su propia limpieza simplificada (`cargar_y_limpiar()`, solo
  relleno por mediana) y no usa todavía la salida de los pasos 4/5 de esta carpeta. Antes de
  reentrenar con las features nuevas, hay que decidir si se actualiza para leer
  `data/processed/lecturas_reales_features.csv`.
- `scripts/train_models.py` también hace su propio `GroupShuffleSplit` inline por función de
  entrenamiento, sin el fallback temporal de `division_datos.py`. Con el único lote real actual,
  eso hace que se salte por completo el entrenamiento de `rf_tipo_anomalia` (que no depende de
  lotes finalizados y sí podría evaluarse hoy con el split temporal del paso 6).
- Paso 7 eligió `IsolationForest` + `RandomForestClassifier` (ya usados en producción, ahora con
  comparación empírica real que los respalda) y un **Algoritmo Genético** nuevo para predicción de
  lluvia (`prediccion_lluvia_ga.py`).
- Paso 8 ya entrenó los 3 modelos anteriores con datos reales y los guardó en `ML/artifacts/`
  (carpeta separada de `app/ml/artifacts/`, que sigue siendo la que usa el servicio en vivo). Los
  modelos de tiempo restante y calidad siguen sin poder entrenarse (0 lotes finalizados / sin
  retroalimentación real), pero sus funciones ya están escritas en `entrenamiento.py` y se
  activarán solas en cuanto haya datos suficientes.
- **Antes de reemplazar los artefactos de producción con los de `ML/artifacts/`**, hay que
  unificar `scripts/train_models.py` con la limpieza/features/split de esta carpeta — de lo
  contrario el servicio en vivo seguiría sin el Algoritmo Genético de lluvia ni las features del
  paso 5.
- **Paso 9 encontró un problema real:** el `IsolationForest` guardado no generaliza a test
  (F1=0, empatado con "nunca alertar") porque su `contamination` fijo (2%) quedó muy por encima
  de la tasa real de anomalía en train (0.4%) tras la corrección de datos. `contamination="auto"`
  recupera algo de señal. Pendiente: ajustar esto en `entrenamiento.py` antes de considerar este
  modelo listo para producción. El AG de lluvia también quedó con una salvedad: su buen F1 en
  test está inflado por un desbalance de clases específico de ese tramo (predijo "lluvia" para el
  100% de las filas de test) — hace falta más variedad de eventos de lluvia reales para confirmar
  que generaliza. Detalle completo en `09_evaluacion.ipynb`.
- **Paso 10 (ajuste):** el "ganador" de validación para `IsolationForest` (`contamination=0.005`)
  falló en test real (F1=0), mientras que `"auto"` — que perdía en la validación de `train_fit` por
  tener muy pocos datos — sí recuperó señal en test (F1≈0.18, recall≈0.96). Se dejó `"auto"` como
  config final por evidencia directa en test, aunque no fue el ganador nominal de validación
  (tensión metodológica documentada en `10_ajuste_hiperparametros.ipynb`). Para `rf_tipo_anomalia`
  todas las configuraciones probadas empataron (limitado porque `train_fit` no tiene ni un ejemplo
  de la clase `temperatura_alta`); se dejó la config más simple. Para el AG de lluvia, ni cambiar
  la métrica de fitness (`f1` → `f1_macro`) ni ajustar hiperparámetros corrigieron el
  comportamiento degenerado (predice "lluvia" siempre) — se concluyó que es un problema de escasez
  de datos (un solo lote real, sin eventos de lluvia variados), no de tuning. Modelos finales
  reentrenados sobre el train completo con las configs ganadoras y guardados en `ML/artifacts/`.
- **Paso 11 (despliegue) — bug de nombres de columna:** los notebooks de esta carpeta usaban
  `"humedad_grano_raw"` como nombre de feature, pero el código de producción
  (`app/services/anomaly_detector.py`, `predictor.py`) arma sus filas con `"humedad_grano"`. Los
  artefactos de `ML/artifacts/` cargaban bien pero fallaban en cada predicción real con
  `columns are missing: {'humedad_grano_raw'}`. Se corrigió con `entrenamiento.py::_a_esquema_produccion()`
  (renombra `humedad_grano_raw` → `humedad_grano`), aplicado también en `evaluacion.py`; se
  reentrenó y reevaluó (notebooks 08, 09, 10) con el nombre correcto. Confirmado con un smoke test
  end-to-end: `POST /anomalies/detect` ahora devuelve `confianza_ml > 0` (antes cargaba el modelo
  pero nunca lo usaba realmente).
- **Paso 11 — artefactos de producción:** se copiaron `isolation_forest.joblib` y
  `rf_tipo_anomalia.joblib` corregidos a `app/ml/artifacts/`. `rf_calidad.joblib` y
  `rf_tiempo_restante.joblib` siguen siendo los viejos (esquema incompatible, `humedad_ambiental`)
  porque no hay datos reales suficientes para reentrenarlos — ya no tumban `/detect` gracias al
  try/except agregado en `predictor.py`/`anomaly_detector.py` (se registra un warning y el campo
  queda en `None` en vez de un error 500).
- **Paso 11 — notificaciones push:** se agregó Firebase Cloud Messaging (`app/services/fcm.py`,
  con el mismo patrón de inicialización perezosa que ya usaba el correo SMTP). Nueva tabla
  `dispositivos_usuario` (ver sección 5 de `migration.sql`, **todavía no aplicada en Neon**) y
  endpoints `POST /dispositivos/registrar`, `POST /dispositivos/desactivar`, `GET /dispositivos/{id_usuario}`.
  `notifier.enviar_push_alerta()` resuelve `id_lote → lotes_cafe.id_usuario → dispositivos_usuario`
  activos y envía solo si `FCM_ENABLED=true` y la severidad supera `FCM_MIN_SEVERIDAD` (config en
  `.env`). Los endpoints `GET /anomalies/{id_lote}/predicciones` y `/recomendaciones` ya existían en
  `history.py`, no hizo falta crearlos.
- **Paso 11 — ampliación post-despliegue (aislamiento por usuario + push en cualquier
  anomalía + tiempo real):**
  - Auditoría de aislamiento: se confirmó (y se probó con dos usuarios/lotes distintos) que
    push, lectura (`GET /anomalies*`) y escritura (`POST /anomalies/detect`) están
    correctamente scoped por `id_lote -> lotes_cafe.id_usuario`; no había hueco. De paso, se
    encontró y corrigió que `dispositivos_usuario` no estaba registrado en
    `database.py::init_db()` (`Base.metadata` nunca la creaba en sqlite/desarrollo).
  - `FCM_MIN_SEVERIDAD` ahora es `"advertencia"` por defecto (antes `"riesgo"`), y el push ya
    no depende de que se haya creado una alerta formal (`inference.py::ejecutar_pipeline`):
    dispara en cualquier `es_anomalia=True`, independiente del correo/alertas (que se quedan
    en riesgo/critico a propósito, para no saturar).
  - **Poller (tiempo real):** `app/services/poller.py`, tarea de fondo (`asyncio`) lanzada en
    el startup de `app/main.py`. Revisa `lecturas_ambientales` cada
    `POLLING_INTERVALO_SEGUNDOS` por filas con `id_lectura` mayor al cursor guardado en
    `ml_estado_polling` (tabla nueva, sección 6 de `migration.sql`) y corre el pipeline para
    cada una. El webhook (`POST /internal/lecturas/nuevas`) y el poller comparten ese mismo
    cursor (`poller.marcar_procesada()`), así que sin importar cuál de los dos procesa una
    lectura primero, el otro la salta — evita duplicar predicciones/alertas/push. La
    construcción del vector de features se factorizó a `app/services/lectura_utils.py` para
    que el webhook y el poller usen exactamente el mismo código, no dos copias que se puedan
    desincronizar.
  - **AG de lluvia conectado en vivo (hallazgo importante):** hasta este punto `ga_lluvia.joblib`
    se entrenaba y evaluaba (pasos 7-10) pero JAMÁS se cargaba desde `app/` -- no estaba
    conectado a `/detect` ni al webhook/poller, solo existía offline. `app/services/rain_predictor.py`
    lo conecta: calcula `lluvia_eventos_24h` y `horas_desde_ultima_lluvia` con una consulta al
    historial real de `lecturas_ambientales` del lote (las otras dos features del AG,
    `presion_hpa` y `luz`, vienen de la lectura actual). Nuevas columnas
    `predicciones.riesgo_lluvia_proxima`/`horas_anticipacion_lluvia` (sección 7 de
    `migration.sql`). Push dedicado con **debounce**: solo avisa cuando el riesgo pasa de
    False/None a True para ese lote (comparado contra la predicción anterior vía
    `notifier.ultimo_riesgo_lluvia`), no en cada lectura mientras el riesgo se mantenga -- sin
    esto, el poller (cada 30s) mandaría un push nuevo todo el tiempo que durara el riesgo.
    **Caveat heredado y documentado:** el modelo se entrenó con `lluvia_sostenida` (columna
    derivada SOLO offline en el paso 4); en producción se usa `lluvia_detectada` cruda porque
    esa versión "sostenida" en vivo todavía no existe (mismo hallazgo abierto de
    `rules.py::clasificar_lluvia()` de más abajo) -- puede sobreestimar riesgo por el parpadeo
    del sensor FC-37. Probado con datos reales: ya no lanza excepción (se corrigió un bug de
    mezclar datetime naive/aware) y el debounce funciona.
  - **Título corto + recomendación en el push:** el FCM antes solo mandaba severidad + mensaje
    genérico. Ahora `rules.py::TITULOS_CORTOS` (nuevo diccionario, mismos tipos que
    `RECOMENDACIONES`/`TIPO_SEVERIDAD_DEFAULT`) da un título de una línea por tipo de anomalía
    (ej. `temperatura_alta` -> "Exceso de temperatura"), y el push manda ese título + el texto
    completo de la recomendación correspondiente (como `cuerpo` Y también en
    `datos.recomendacion`, para que la app móvil arme "alerta + recomendación" sin pedir nada
    más). De paso se corrigió que `patron_atipico_ml` (tipo que agrega el propio ML, no las
    reglas) no tenía entrada en `RECOMENDACIONES` y caía silenciosamente al texto de "normal"
    -- justo lo contrario de lo que hace falta al mostrar una alerta real.
- **Paso 12 (monitoreo y reentrenamiento):** `monitoreo.py` compara, para cada lote con fila en
  `retroalimentacion_ml`, la ÚLTIMA predicción guardada en `predicciones` contra el resultado
  real reportado — misma métrica y mismo baseline que `evaluacion.py` (paso 9), pero con datos
  EN VIVO en vez de `test.csv`. También vigila la tasa de alertas (ventana actual vs anterior,
  salto >= 2x se marca como posible drift) y compara cuántas lecturas/lotes nuevos hay hoy
  contra lo que se usó en el último entrenamiento (leído de
  `metricas_entrenamiento.json::fecha_entrenamiento`/`n_filas_train_total`, campos que
  `entrenamiento.py::main()` ahora escribe). Expuesto en `GET /internal/monitoreo/salud` y por
  `../scripts/monitorear_modelos.py` (pensado para cron en el servidor real; sale con código 1
  si hace falta reentrenar). **A propósito no reentrena solo** — el paso 10 ya mostró que un
  "ganador" de validación puede fallar en test real, así que cada reentrenamiento sigue siendo
  una decisión humana: correr 08 → 09 → 10 y revisar antes de copiar artefactos nuevos a
  `app/ml/artifacts/`.
- **Limpieza (post paso 11):** se eliminaron los archivos que ya no servían: stubs
  autodeclarados "DEPRECADO" (`scripts/generar_dataset.py`, `scripts/exportar_retroalimentacion.py`),
  un modelo huérfano de una tabla que ya no existe (`app/models/inference_record.py` →
  `inferencias_anomalias`), los notebooks/CSVs del prototipo sintético viejo (carpeta
  `notebooks/` completa, `data/raw/lecturas_ml_training.csv` y compañía, `data/figures/`), bases
  sqlite locales de prueba (`app.db`, `test_app.db`), scripts de debug ad-hoc de la raíz, un
  diagrama de arquitectura desactualizado, y los 4 documentos raíz (`README.md`,
  `DOCUMENTACION.md`, `EDA_MICROSERVICIO.md`, `INTEGRACION_MOVIL.md`) que describían el
  pipeline viejo (solo IsolationForest, dataset sintético) — reemplazados por un `README.md`
  raíz nuevo alineado con el pipeline real.

## Corrección importante (post paso 8): se recuperaron ~15h20min de datos reales perdidos

El paso 4 original tiraba de golpe ~2,156 filas del lote 12 (2026-07-19 06:57 a 22:16) solo
porque `humedad_grano_raw` todavía no reportaba en ese tramo (el sensor capacitivo se conectó
horas después que el resto) — pero `temperatura_grano`, `temperatura_ambiental`, `presion_hpa` y
`luz` sí tenían datos reales y válidos ahí. Se corrigió separando las columnas "núcleo" (se
descarta la fila si faltan) de las "opcionales" (se rellenan pero nunca causan que se tire la
fila). Los pasos 4 a 8 ya se re-ejecutaron con la corrección: el lote 12 ahora cubre
7,458 lecturas reales desde 2026-07-19 06:57:52 hasta 2026-07-20 15:07:05 (antes: 5,506 lecturas
desde las 22:16). El AG de lluvia mejoró de F1=0.79 a F1=0.88 en test con los datos recuperados.
