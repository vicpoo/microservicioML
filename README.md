<!-- Archivo: README.md -- Carpeta: microservicioMLL/ (raíz del proyecto) -->

# microservicioMLL v2

Microservicio de detección de anomalías, predicciones y recomendaciones para el secado de café,
conectado a la base de datos de producción en Neon (Postgres).

## Qué cambió respecto a la v1

- **Base de datos**: ya no usa SQLite local (`app.db`) ni una tabla propia `inferencias_anomalias`.
  Ahora se conecta a tu Neon y usa las tablas que ya existen ahí: `alertas`, `predicciones`,
  `recomendaciones`, `inferencias_ml`, `modelos_ml`, `sensores`, `lotes_cafe`, `lecturas_ambientales`.
- **Sensores reales**: se eliminó `viento` (no hay anemómetro en tu kit IoT: BME280, DS18B20,
  BH1750, FC-37, sensor de humedad de grano). Ahora son 6 variables: `temperatura_grano`,
  `temperatura_ambiental`, `humedad_ambiental`, `humedad_grano`, `lluvia`, `luz`.
- **Detección de anomalías**: ensamble de reglas de dominio (umbrales del PDF de calidad,
  `app/services/rules.py`) + `RandomForestClassifier` (generaliza patrones) + `IsolationForest`
  (atrapa outliers que ni las reglas ni el clasificador conocen).
- **Predicciones**: `RandomForestRegressor` para horas restantes de secado y
  `RandomForestClassifier` para calidad final estimada (excelente/buena/regular/baja).
- **Recomendaciones**: texto accionable por tipo de alerta (Cuadro 10 del documento de dominio),
  se guarda en `recomendaciones`.
- **Notificaciones**: cada alerta se guarda en `alertas` (para que la app la consuma) y,
  si `EMAIL_ENABLED=true` y hay credenciales SMTP, se envía correo para severidad `riesgo`/`critico`.

## 0. Arquitectura (el MLL es 100% interno, no habla con la app)

```
ESP32 --MQTT--> Servicio Gestor --guarda--> Neon Postgres <---- microservicioMLL
                       |                    (alertas, predicciones,    |  ^
                       |                     recomendaciones, etc.)    |  |
                       |--publica--> WebSocket -----> App móvil        |  |
                       '--ping (X-Internal-Api-Key)--------------------'  |
                                                                            |
                                                    API móvil (login) -----'
                                                        |
                                                        '--> App móvil (alertas, predicciones, recomendaciones)
```

El MLL nunca le contesta nada a la app ni valida sesiones de usuario. Su trabajo es:

1. **Leer**: el Gestor, justo después de guardar una lectura en `lecturas_ambientales`, le avisa
   al MLL (`POST /api/v1/internal/lecturas/nuevas`, header `X-Internal-Api-Key`) "hay algo nuevo
   en el lote X". El MLL va y lee el dato real de Neon (no depende del formato que mande el
   Gestor) y corre el pipeline (reglas + RandomForest + IsolationForest).
2. **Escribir**: guarda alertas, predicciones y recomendaciones directo en las tablas de Neon
   (`alertas`, `predicciones`, `recomendaciones`, más la bitácora en `inferencias_ml`).
3. **Tu API móvil** (la que ya maneja login y datos de usuario) lee esas mismas tablas de Neon
   y se las entrega a la app — no necesita pasar por el MLL para eso, ya están ahí guardadas.

Por si en algún momento prefieres que tu API móvil le pida el historial al MLL en vez de leer
Neon directo, dejé los endpoints `GET /api/v1/anomalies`, `.../predicciones`, `.../recomendaciones`
disponibles (ver más abajo) — protegidos con la misma `X-Internal-Api-Key`, no son de uso
obligatorio.

`POST /api/v1/anomalies/detect` (recibe las 6 lecturas directo en el body) se deja disponible
para pruebas manuales / curl, también protegido con `X-Internal-Api-Key`; en producción el
disparador real es el endpoint interno del punto 1.

**Una sola capa de autenticación**: `X-Internal-Api-Key`, compartida entre el MLL y los servicios
internos tuyos que lo llamen (Gestor, y opcionalmente tu API móvil). Como el MLL ya no habla con
la app ni con usuarios finales, no necesita validar JWT.

**Alertas vs. predicciones/recomendaciones**: las predicciones (horas restantes, calidad) y las
recomendaciones se generan con CADA lectura nueva, usando los 6 sensores. Las alertas (tabla
`alertas` + correo) solo se disparan cuando la severidad es `riesgo` o `critico` — una
`advertencia` leve queda solo en las predicciones/recomendaciones, sin "molestar" al usuario.

Los modelos de ML en sí no tienen problema de mezcla de datos entre usuarios: cada predicción es
sin estado, solo usa los datos que le mandas en esa llamada puntual.

## 1. Migrar la base de datos

**Ya aplicada** — confirmado contra tu respaldo más reciente (`temperatura_grano`, `luz` y
`lluvia` ya existen en `lecturas_ambientales`). No hace falta correr `migration.sql` de nuevo;
se deja en el proyecto solo como referencia/histórico.

## 2. Configurar variables de entorno

Copia `.env.example` a `.env` y ajusta:

```bash
cp .env.example .env
```

- `DATABASE_URL`: ya viene con tu cadena de Neon (driver `psycopg2`).
- `EMAIL_ENABLED` / `SMTP_*` / `ALERT_EMAIL_TO`: para que las alertas `riesgo`/`critico` también
  lleguen por correo. Si usas Gmail necesitas una "contraseña de aplicación", no tu contraseña normal.
  Si lo dejas en `false`, las alertas se siguen guardando en la tabla `alertas` (la app las puede leer),
  solo no se manda correo.

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 4. (Re)generar el dataset sintético y entrenar los modelos

Ya vienen artefactos entrenados en `app/ml/artifacts/`, pero si quieres regenerarlos (por ejemplo,
después de ajustar los umbrales en `app/services/rules.py`):

```bash
python scripts/generar_dataset.py   # crea data/raw/lecturas_ml_training.csv
python scripts/train_models.py      # entrena y guarda los 4 artefactos .joblib
```

`scripts/train_models.py` acepta un argumento opcional para entrenar solo un modelo
(`isolation_forest`, `tipo`, `tiempo` o `calidad`), útil si el entrenamiento completo tarda mucho
en tu máquina.

**Nota honesta sobre las métricas actuales** (dataset 100% sintético, ver `app/ml/artifacts/metricas.json`):
clasificación de tipo de anomalía ~87% accuracy, calidad final ~41% accuracy (predecir la calidad
final de un lote a partir de una sola lectura es difícil; mejora mucho si luego reentrenas con lotes
reales ya finalizados). Recalibrar contra datos reales de campo es el siguiente paso natural.

## 5. Correr el servicio

```bash
uvicorn app.main:app --reload
```

## Endpoint principal

`POST /api/v1/anomalies/detect`

```json
{
  "id_usuario": 7,
  "id_lote": 1,
  "tipo_proceso": "lavado",
  "id_sensor": 1,
  "lecturas": {
    "temperatura_grano": 30.0,
    "temperatura_ambiental": 26.0,
    "humedad_ambiental": 70.0,
    "humedad_grano": 40.0,
    "lluvia": 0.9,
    "luz": 5000
  }
}
```

Si mandas `id_lote` y el lote existe en `lotes_cafe`, el servicio:
1. Toma `tipo_proceso` y calcula horas transcurridas desde `fecha_inicio_secado`.
2. Guarda la lectura cruda en `lecturas_ambientales` (si conoce el sensor).
3. Corre el ensamble de detección de anomalías y el predictor.
4. Si hay alerta (`riesgo`/`critico`/`advertencia`), la guarda en `alertas` y, si aplica, manda correo.
5. Guarda recomendaciones en `recomendaciones` y la predicción en `predicciones`.
6. Registra la inferencia completa en `inferencias_ml` (bitácora, siempre, con o sin `id_lote`).

Si NO mandas `id_lote` (o no existe), igual te devuelve el análisis completo, solo que no persiste en
`alertas`/`recomendaciones`/`predicciones` (esas tablas exigen `id_lote`), solo en la bitácora.

### Endpoint interno (lo llama el Gestor, no la app)

`POST /api/v1/internal/lecturas/nuevas` con header `X-Internal-Api-Key`:
```json
{"id_lote": 1, "id_lectura": 4821}
```
`id_lectura` es opcional; si no lo mandas, toma la lectura más reciente de ese lote.

### Endpoints opcionales de historial (por si tu API móvil los quiere usar)

Todos con header `X-Internal-Api-Key` + `id_usuario` como query param (lo resuelve quien llama,
normalmente tu API móvil ya sabe qué usuario está pidiendo):

- `GET /api/v1/anomalies?id_lote=1&id_usuario=7` — historial de alertas.
- `GET /api/v1/anomalies/1/predicciones?id_usuario=7` — historial de predicciones del lote 1.
- `GET /api/v1/anomalies/1/recomendaciones?id_usuario=7` — historial de recomendaciones del lote 1.
- `GET /health` (sin auth)

De nuevo: **no es obligatorio usarlos**. Como el MLL ya escribió alertas/predicciones/recomendaciones
directo en Neon, tu API móvil puede simplemente hacer sus propios `SELECT` a esas tablas si le
resulta más simple que llamar al MLL.

## Pruebas

```bash
DATABASE_URL="sqlite:////tmp/app_test.db" pytest tests/ -v
```

(usa sqlite local para no tocar tu Neon de producción durante las pruebas).

## Estructura

```
app/
  core/config.py          # variables de entorno (DB, SMTP)
  models/                 # ORM: mapea las tablas YA existentes en Neon
  schemas/                 # request/response de la API
  services/
    rules.py               # umbrales de dominio (única fuente de verdad, la usan generador y detector)
    preprocessor.py         # arma el vector de 6 features
    anomaly_detector.py      # reglas + RandomForest + IsolationForest
    predictor.py             # RandomForest: horas restantes + calidad estimada
    recommender.py           # texto de recomendación por tipo de alerta
    notifier.py               # persiste alertas/recomendaciones/predicciones + envía correo
  api/routes/inference.py, history.py
  ml/artifacts/            # modelos entrenados (.joblib)
scripts/
  generar_dataset.py       # dataset sintético (reglas de dominio + simulación física por lote)
  train_models.py          # entrena los 4 modelos
migration.sql              # migración para Neon (columnas nuevas + seed de modelos_ml)