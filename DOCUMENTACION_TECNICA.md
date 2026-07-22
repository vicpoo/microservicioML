# Documentación técnica — microservicioMLL (kajve)

Referencia completa para conectar el frontend/app móvil y el Servicio Gestor con este
microservicio: qué hace cada modelo (ML, Algoritmo Genético, NLP, NLG), qué endpoints existen,
el JSON exacto que cada uno recibe y devuelve, cómo activar FCM en producción, y cuánto CPU/RAM
necesita para correr bien.

---

## 1. Arquitectura general

```
ESP32 (sensores) --> Servicio Gestor (backend principal, dueño de Neon)
                            |
                            |  POST /internal/lecturas/nuevas   (webhook, instantáneo)
                            v
                    microservicioMLL (este servicio)
                            |
              +-------------+--------------+
              |                            |
        Poller interno              Pipeline de inferencia
        (cada 30s, red de           (ML + AG + reglas)
         seguridad si el                   |
         webhook falla)             +------+------+
                                    |             |
                              Persiste en    Notifica
                              Neon (alertas,  (FCM push,
                              predicciones,   correo)
                              recomendaciones)
                                    |
                            App móvil / frontend
                            (lee historial + PLN
                             vía GET, o directo
                             de Neon)
```

El microservicio es **100% interno**: no valida JWT de usuarios finales ni habla directo con la
app móvil. Todo pasa por el Servicio Gestor (o, si tu arquitectura lo prefiere, tu API móvil
puede llamarlo directo, pero sigue siendo tráfico "de confianza" autenticado con un API key
compartido, no con el login del usuario final).

**Aislamiento por usuario:** cada sensor pertenece a un lote, cada lote a un usuario
(`id_sensor -> lotes_cafe.id_lote -> lotes_cafe.id_usuario -> dispositivos_usuario.id_usuario`).
Todo endpoint de lectura/escritura valida esta cadena en código (no hay RLS de Postgres que lo
haga solo) — un usuario nunca puede leer ni recibir notificaciones de datos de otro.

---

## 2. Stack tecnológico

| Pieza | Tecnología |
|---|---|
| Framework HTTP | FastAPI + Uvicorn |
| ORM / BD | SQLAlchemy, Postgres en Neon (producción) / SQLite (dev y tests) |
| ML clásico | scikit-learn (`IsolationForest`, `RandomForestClassifier`) + `joblib` para servir |
| Algoritmo Genético | NumPy puro (implementación propia, sin librería de AG externa) |
| NLP — ranking/búsqueda | `rank_bm25` (BM25 Okapi) |
| NLP — clasificación | scikit-learn (`TfidfVectorizer` + `MultinomialNB`) |
| NLG | Plantillas Python puras (sin librería externa) |
| Notificaciones push | `firebase-admin` (FCM) |
| Notificaciones correo | `smtplib` (estándar de Python) |
| Tiempo real | `asyncio` (poller en segundo plano dentro del mismo proceso) |
| Pruebas | `pytest` + `TestClient` de FastAPI |

---

## 3. Modelos de Machine Learning

### 3.1 Detección de anomalías — `IsolationForest`

No supervisado. Aprende qué combinaciones de temperatura/humedad/lluvia/luz son "normales" por
tipo de proceso (lavado/honey/natural) y marca como outlier lo que se desvía, sin que nadie le
diga de antemano qué es una anomalía. Se combina con un motor de reglas de umbral explícito
(`app/services/rules.py`, basado en el documento de calidad del café) — si ninguna regla cubre
el patrón pero el IsolationForest sí lo marca raro, se etiqueta `patron_atipico_ml`.

- Archivo servido: `app/ml/artifacts/isolation_forest.joblib` (~1.8 MB)
- Features: `temperatura_grano, temperatura_ambiental, humedad_grano, lluvia, luz, delta_temp`
- Entrenamiento: `ML/entrenamiento.py`, sobre datos reales de Neon (`scripts/recolectar_datos_reales.py`)

### 3.2 Clasificación del tipo de anomalía — `RandomForestClassifier`

Dado que sí hay anomalía, predice cuál: `temperatura_alta`, `lluvia_detectada`,
`secado_estancado`, `fluctuacion_termica`, `radiacion_insuficiente`, `valor_imposible`, o
`patron_atipico_ml` (cuando el IsolationForest generaliza algo que ninguna regla cubre).

- Archivo servido: `app/ml/artifacts/rf_tipo_anomalia.joblib` (~188 KB)

### 3.3 Tiempo restante y calidad final — **no disponibles todavía**

Dos `RandomForest` (regresor y clasificador) iban a predecir horas restantes de secado y
calidad final. Se eliminaron los artefactos que existían (estaban entrenados con un dataset
sintético viejo, esquema incompatible con el hardware real) y hoy `predictor.py` responde
`tiempo_estimado_horas` / `calidad_estimada` en `null` hasta que se acumulen suficientes lotes
reales terminados (`retroalimentacion_ml`) y se corra `scripts/train_models.py`. `GET
/internal/monitoreo/salud` dice cuándo ya hay datos suficientes.

### 3.4 Algoritmo Genético — riesgo de lluvia próxima

Predice **riesgo de que llueva en las próximas H horas** (no "está lloviendo ahora mismo", eso
ya lo resuelve directo el sensor FC-37). Implementado en `ML/prediccion_lluvia_ga.py`, servido
en vivo por `app/services/rain_predictor.py`.

**Por qué un AG y no una red neuronal:** con solo 2 sensores relevantes disponibles (presión
BMP280, luz BH1750, sin estación meteorológica), el espacio de variables es chico. El resto del
sistema ya es reglas de umbral 100% interpretables — un AG sigue ese mismo espíritu: en vez de
un experto adivinando pesos a mano, el AG los evoluciona optimizando F1 directamente sobre datos
reales, pero el resultado sigue siendo una regla lineal con pesos explícitos, auditable, no una
caja negra.

**Cómo funciona:**
- Individuo = vector `[peso_presión, peso_luz, peso_eventos_lluvia_24h, peso_horas_desde_última_lluvia, bias]`
- Predicción = `score = pesos · variables_normalizadas + bias`; si `score > 0` → riesgo de lluvia
- Fitness = F1-score "macro" (promedio entre ambas clases — con F1 simple el AG aprendió a
  hacer trampa prediciendo "va a llover" siempre; se corrigió tras encontrarlo en evaluación)
- Evolución: 60 individuos, 40 generaciones, elitismo (el mejor pasa intacto), selección por
  torneo (3 individuos al azar, gana el de mejor fitness), cruza uniforme, mutación gaussiana
- Archivo servido: `app/ml/artifacts/ga_lluvia.joblib` (~1.7 KB)
- Predicción por defecto: 3 horas de anticipación (`horas_anticipacion_lluvia`)

**Limitación conocida:** se entrenó con una señal de "lluvia sostenida" que hoy solo existe
offline; en producción usa el sensor crudo, lo que puede sobreestimar el riesgo por el parpadeo
conocido del sensor FC-37.

---

## 4. NLP y NLG

Viven en la carpeta `NLP/`, separada de `ML/` a propósito (dos técnicas de IA distintas para
problemas distintos). **No es un chatbot ni nada conversacional.**

### 4.1 NLG por plantillas — reportes en lenguaje natural

`NLP/recopilar_datos_reporte.py` + `NLP/generar_reporte.py`: convierten los datos ya generados
por ML (alertas, predicciones, recomendaciones) en un texto en español, 100% determinístico y
auditable (sin modelo, sin generación probabilística — mismos datos, mismo texto siempre).
Ejemplo real generado:

> "El lote kajve-CA689C (proceso lavado) lleva 32 horas y 9 minutos en secado. Se registraron 3
> alertas: 2 de exceso de temperatura, 1 de lluvia detectada. La más reciente fue de exceso de
> temperatura y ya fue atendida. El modelo no estima riesgo de lluvia en las próximas 3 horas.
> Recomendaciones: (1) Mueve el lote a sombra parcial... (2) Prioridad máxima: cubre el lote..."

### 4.2 BM25 — resumen extractivo (dentro de un reporte)

`NLP/rankear_eventos.py`. Cuando un lote acumula más de 5 alertas, en vez de solo contar tipos,
BM25 puntúa cada mensaje de alerta contra un query fijo de términos de urgencia
("crítico", "riesgo", "lluvia", "estancado"...) y selecciona los 3 más relevantes — resumen
extractivo real (selecciona texto que ya existe, no genera nada nuevo).

### 4.3 BM25 — buscador de historial (entre TODOS los reportes de un usuario)

`NLP/buscar_reportes.py`. Mismo algoritmo, usado como lo que realmente es: un motor de búsqueda
de texto libre sobre todo el historial de reportes ya generados de un usuario (todos sus lotes).
Con corpus muy chicos (1-2 reportes, el caso más común de un usuario nuevo) BM25 es
matemáticamente degenerado, así que hay un fallback a conteo simple de coincidencia de palabras
para esos casos — documentado en el propio código.

**Por qué BM25 y no BM42:** BM42 necesita una base de datos vectorial + un modelo de embeddings
(infraestructura extra que no se justifica aquí). BM25 es puro Python, sin servicios externos,
auditable, y suficiente para el volumen de texto de este proyecto.

### 4.4 TF-IDF + Naive Bayes — clasificador de texto ligero

`NLP/entrenar_clasificador_texto.py` (entrenamiento) + `NLP/clasificar_texto.py` (servir).
Clasifica texto libre en español y sugiere severidad (`alta`/`critica`) usando
`TfidfVectorizer` + `MultinomialNB`, entrenado sobre `alertas.mensaje`/`alertas.nivel_severidad`.

**Aviso importante:** los mensajes de alerta actuales salen de plantillas fijas (mismo texto
siempre por tipo), así que el modelo memoriza esos strings casi perfecto (exactitud ~100% sobre
su propio entrenamiento) sin que eso signifique que "entiende español". El valor real de esta
pieza es quedar lista para el día en que exista texto libre genuino que clasificar — por ejemplo
notas escritas a mano por el productor (ese campo no existe todavía en el esquema).

Solo entrena si hay al menos 10 alertas reales acumuladas (`MIN_EJEMPLOS_ENTRENAMIENTO`); si no,
el endpoint responde `disponible: false` en vez de dar una predicción inventada.

---

## 5. Tiempo real: cómo se dispara el pipeline

Dos caminos, ambos comparten un cursor en BD (`ml_estado_polling`) para no procesar la misma
lectura dos veces:

1. **Webhook (preferido):** el Gestor llama `POST /internal/lecturas/nuevas` apenas guarda una
   lectura nueva en Neon. El microservicio va y la lee él mismo (el Gestor no necesita saber
   nada del formato del modelo).
2. **Poller (red de seguridad):** tarea de fondo (`asyncio`) que revisa `lecturas_ambientales`
   cada `POLLING_INTERVALO_SEGUNDOS` (default 30) por si el webhook falló. Se puede apagar con
   `POLLING_ENABLED=false` si se prefiere 100% reactivo.

---

## 6. Endpoints — referencia completa

Todos los endpoints (excepto `/health`) requieren el header `X-Internal-Api-Key` si
`INTERNAL_API_KEY` está configurado en el `.env` del servidor (en desarrollo local, vacío = sin
exigencia). Prefijo base: `/api/v1` (excepto `/health`).

### 6.1 `POST /api/v1/anomalies/detect` — detección manual / pruebas

Uso: pruebas manuales (curl, Postman) o si tu frontend quiere mandar lecturas directo sin pasar
por el Gestor. En producción normal, el disparador real es 6.2.

**Request:**
```json
{
  "id_usuario": 1,
  "id_lote": 42,
  "tipo_proceso": "lavado",
  "id_sensor": 7,
  "timestamp": "2026-07-22T10:00:00Z",
  "lecturas": {
    "temperatura_grano": 38.5,
    "temperatura_ambiental": 26.0,
    "humedad_grano": 2100,
    "lluvia": 0.0,
    "luz": 42000,
    "presion_hpa": 1013.2
  },
  "guardar_lectura": true
}
```
- `id_lote`, `tipo_proceso`, `id_sensor`, `timestamp` son opcionales (si mandas `id_lote`, el
  tipo de proceso se toma del lote y se ignora el que mandes).
- `presion_hpa` es opcional dentro de `lecturas`: si se manda, activa la predicción de lluvia
  del AG; si no, esa predicción queda en `null` y el resto del pipeline sigue igual.

**Response 200:**
```json
{
  "id_inferencia": 501,
  "id_lote": 42,
  "es_anomalia": true,
  "nivel_severidad": "critico",
  "score_isolation_forest": -0.18,
  "confianza_ml": 0.82,
  "variables_contribuyentes": ["temperatura_grano"],
  "mensaje": "Riesgo crítico: se requiere atención inmediata. Temperatura del grano por encima del rango ideal para este proceso.",
  "recomendaciones": [
    {"tipo": "temperatura_alta", "texto": "Mueve el lote a sombra parcial o reduce su exposición directa al sol..."}
  ],
  "prediccion": {
    "tiempo_estimado_horas": null,
    "calidad_estimada": null,
    "confianza": null,
    "riesgo_lluvia_proxima": false,
    "horas_anticipacion_lluvia": 3
  },
  "alerta_generada": true,
  "id_alerta": 88,
  "notificacion_email_enviada": false,
  "notificacion_push_enviada": true,
  "modelo_version": "2.0.0",
  "fecha_inferencia": "2026-07-22T10:00:01.123456Z"
}
```
`nivel_severidad`: `normal | advertencia | riesgo | critico`.

### 6.2 `POST /api/v1/internal/lecturas/nuevas` — webhook del Gestor (producción)

**Request:**
```json
{ "id_lote": 42, "id_lectura": 9001 }
```
`id_lectura` es opcional (si no se manda, toma la más reciente del lote). El microservicio lee
la fila real de `lecturas_ambientales` en Neon y corre el mismo pipeline que 6.1.

**Response 200:** misma forma que 6.1 (`InferenceResponse`).

### 6.3 `POST /api/v1/internal/lotes/{id_lote}/resultado-real` — retroalimentación real

Se llama cuando el productor reporta el resultado final de un lote (para reentrenar con datos
reales, no sintéticos).

**Request:**
```json
{ "calidad_real": "buena", "tiempo_real_horas": 180.5 }
```
`calidad_real`: `excelente | buena | regular | baja`. `tiempo_real_horas` es opcional (si no se
manda, se calcula desde `fecha_inicio_secado` del lote hasta ahora).

**Response 201:**
```json
{ "id_retroalimentacion": 12, "mensaje": "Resultado real registrado" }
```

### 6.4 `GET /api/v1/internal/monitoreo/salud` — salud del modelo (paso 12)

Query param opcional: `dias_alertas` (default 7). Pensado para un cron/dashboard, no para la app
móvil.

**Response 200 (forma resumida):**
```json
{
  "fecha_reporte": "2026-07-22T10:00:00Z",
  "reentrenamiento": {
    "necesita_reentrenamiento": false,
    "razones": [],
    "disponibilidad_datos": {
      "filas_nuevas_desde_ultimo_entrenamiento": 340,
      "umbral_filas_nuevas": 2000,
      "lotes_faltantes_para_tiempo_restante": 4,
      "lotes_faltantes_para_calidad": 4
    },
    "desempeno_produccion": { "omitido": "retroalimentacion_ml está vacía todavía..." }
  },
  "monitoreo_alertas": {
    "ventana_dias": 7,
    "alertas_ventana_actual": 12,
    "alertas_ventana_anterior": 9,
    "razon_actual_vs_anterior": 1.33,
    "posible_drift": false
  }
}
```

### 6.5 `GET /api/v1/anomalies` — historial de alertas

Query params: `id_usuario` (requerido), `id_lote` (opcional), `solo_no_atendidas` (bool),
`limit` (default 20), `offset`.

**Response 200:**
```json
[
  {
    "id_alerta": 88, "id_lote": 42, "id_sensor": 7,
    "tipo_alerta": "temperatura_alta", "mensaje": "Riesgo crítico: ...",
    "nivel_severidad": "critica", "atendida": false,
    "fecha_generada": "2026-07-22T10:00:01"
  }
]
```
`nivel_severidad` aquí usa la escala de 4 niveles de la BD (`baja/media/alta/critica`), distinta
de la escala del motor de reglas (`normal/advertencia/riesgo/critico`) — ver mapeo en
`app/services/notifier.py::_SEVERIDAD_A_NIVEL`.

### 6.6 `GET /api/v1/anomalies/{id_lote}/predicciones` — historial de predicciones

Query: `id_usuario` (requerido), `limit`.
```json
[
  {
    "id_prediccion": 301, "id_lote": 42,
    "tiempo_estimado_horas": null, "calidad_estimada": null, "confianza": null,
    "fecha_prediccion": "2026-07-22T10:00:01"
  }
]
```

### 6.7 `GET /api/v1/anomalies/{id_lote}/recomendaciones` — historial de recomendaciones

```json
[
  { "id_recomendacion": 501, "id_lote": 42, "texto": "Mueve el lote a sombra parcial...",
    "origen": "modelo_ml", "fecha_generada": "2026-07-22T10:00:01" }
]
```

### 6.8 `GET /api/v1/anomalies/{id_lote}/reporte` — reporte NLG (genera y guarda)

Query: `id_usuario` (requerido).
```json
{
  "id_reporte": 15,
  "id_lote": 42,
  "reporte_texto": "El lote kajve-CA689C (proceso lavado) lleva 32 horas...",
  "fecha_generado": "2026-07-22T10:00:01.581327"
}
```
Cada llamada genera el texto AL MOMENTO y lo guarda (no se sobrescribe el historial).

### 6.9 `GET /api/v1/anomalies/{id_lote}/reportes` — historial de reportes

Query: `id_usuario`, `limit` (default 10).
```json
[
  { "id_reporte": 15, "id_lote": 42, "reporte_texto": "...", "fecha_generado": "..." },
  { "id_reporte": 12, "id_lote": 42, "reporte_texto": "...", "fecha_generado": "..." }
]
```

### 6.10 `GET /api/v1/anomalies/reportes/buscar` — buscador BM25 de historial

Query: `id_usuario` (requerido), `query` (texto libre, requerido), `top_n` (default 5).
```
GET /api/v1/anomalies/reportes/buscar?id_usuario=1&query=lluvia%20critica&top_n=5
```
```json
[
  {
    "id_reporte": 15, "id_lote": 42, "score": 0.624,
    "reporte_texto": "El lote ... lluvia detectada ...",
    "fecha_generado": "2026-07-22T10:00:01"
  }
]
```
Regresa `[]` (200, no error) si no hay coincidencia real o el usuario no tiene reportes.

### 6.11 `POST /api/v1/nlp/clasificar-texto` — clasificador de texto ligero

**Request:**
```json
{ "id_usuario": 1, "texto": "el productor escribió que hay agua encima del café" }
```

**Response 200 (modelo disponible):**
```json
{
  "disponible": true,
  "severidad_sugerida": "critica",
  "confianza": 78.9,
  "probabilidades": { "alta": 21.1, "critica": 78.9 }
}
```
**Response 200 (modelo no entrenado todavía):**
```json
{ "disponible": false, "mensaje": "El clasificador todavía no está disponible..." }
```

### 6.12 Dispositivos (registro de tokens FCM)

**`POST /api/v1/dispositivos/registrar`** — llamar cuando la app móvil obtiene/renueva su token
FCM (típicamente al iniciar sesión o al arrancar la app):
```json
{ "id_usuario": 1, "fcm_token": "eaX...token-largo-de-firebase...", "plataforma": "android" }
```
`plataforma`: `android | ios | web`. Es upsert: si el token ya existía para ese usuario, se
reactiva en vez de duplicar.

Response 201:
```json
{
  "id_dispositivo": 5, "id_usuario": 1, "plataforma": "android",
  "activo": true, "fecha_registro": "2026-07-22T09:00:00",
  "fecha_ultima_actualizacion": "2026-07-22T09:00:00"
}
```

**`POST /api/v1/dispositivos/desactivar`** — al hacer logout/desinstalar:
```json
{ "id_usuario": 1, "fcm_token": "eaX...token-largo..." }
```
Response 200: `{ "desactivado": true }`

**`GET /api/v1/dispositivos/{id_usuario}`** — diagnóstico, lista dispositivos activos:
```json
[{ "id_dispositivo": 5, "id_usuario": 1, "plataforma": "android", "activo": true, ... }]
```

### 6.13 `GET /health` — liveness check (sin auth, sin prefijo `/api/v1`)

```json
{ "status": "ok", "service": "microservicioMLL", "modelo_version": "2.0.0" }
```

---

## 7. Cómo conectar tu frontend

Depende de qué tan directo quieras que hable con el microservicio:

- **Opción recomendada (vía Servicio Gestor):** tu frontend nunca llama al MLL directo. Manda
  todo a tu API principal (Gestor), y el Gestor:
  - reenvía las lecturas del ESP32 y llama `POST /internal/lecturas/nuevas`,
  - reenvía el token FCM del celular a `POST /dispositivos/registrar` cuando el usuario hace login,
  - expone tú mismo el historial/reportes a la app leyendo Neon directo, o pidiéndoselo al MLL
    con los GETs de la sección 6 (5-11) y reenviando la respuesta tal cual.
- **Opción directa:** tu frontend (o tu backend intermedio) llama directo a los GETs de
  historial/reportes/búsqueda/clasificador con el header `X-Internal-Api-Key`. Nunca expongas
  ese API key en el código del cliente móvil directamente — debe vivir en tu backend, no en la
  app.

Para pintar la pantalla de un lote en la app, el orden típico de llamadas es:
1. `GET /anomalies?id_lote=X&id_usuario=Y` — alertas
2. `GET /anomalies/{id_lote}/predicciones?id_usuario=Y` — predicciones (tiempo/calidad/lluvia)
3. `GET /anomalies/{id_lote}/reporte?id_usuario=Y` — un solo texto legible con todo lo anterior
   ya resumido (evita tener que armar la UI a partir de 3 respuestas distintas)

Para notificaciones push, el frontend solo necesita:
1. Pedir permiso de notificaciones y obtener el token FCM (SDK de Firebase en la app).
2. Mandar ese token a tu backend, que lo reenvía a `POST /dispositivos/registrar`.
3. Ya no hace falta hacer nada más: el MLL manda el push automáticamente cuando detecta una
   anomalía (ver sección 8).

---

## 8. Conectar FCM para producción

Hoy `FCM_ENABLED=false` por default — las notificaciones push son un no-op silencioso (no
truena nada, simplemente no manda). Para activarlas en producción:

1. Ve a [Firebase Console](https://console.firebase.google.com) y crea un proyecto (o usa uno
   ya existente si tu app móvil ya está registrada en Firebase).
2. **Configuración del proyecto > Cuentas de servicio > Generar nueva clave privada** — descarga
   el archivo JSON del service account.
3. Sube ese JSON al servidor donde corre el microservicio (**nunca lo subas al repositorio de
   código** — agrégalo a `.gitignore` si no está ya).
4. En el `.env` del servidor:
   ```
   FCM_ENABLED=true
   FCM_CREDENTIALS_PATH=/ruta/absoluta/al/archivo-service-account.json
   FCM_MIN_SEVERIDAD=advertencia
   ```
   `FCM_MIN_SEVERIDAD` controla desde qué severidad se manda push: `advertencia` (default) avisa
   en CUALQUIER anomalía detectada; `riesgo` o `critico` solo en las más graves.
5. En tu app móvil (Android/iOS/web), integra el SDK de Firebase Cloud Messaging normal (esto es
   del lado del cliente, no de este microservicio) para obtener el token FCM del dispositivo.
6. Cada vez que la app obtenga/renueve ese token, mándalo a tu backend, que lo reenvía a
   `POST /api/v1/dispositivos/registrar` (ver sección 6.12). Sin esto, el MLL no tiene a quién
   mandarle la notificación aunque FCM esté bien configurado (siempre filtra por
   `dispositivos_usuario` del dueño real del lote).
7. Reinicia el servicio para que tome la nueva configuración.

**Cómo se arma cada notificación:** título corto (ej. "Exceso de temperatura", "Lluvia
detectada"), cuerpo = la recomendación completa ya redactada, y un payload `data` con
`{"tipo", "recomendacion", "id_lote", "severidad"}` para que la app pueda mostrar "alerta +
recomendación" en una sola vista sin llamar a otro endpoint. El riesgo de lluvia del AG manda su
propio push, mostrado solo cuando el riesgo pasa de "no" a "sí" (no se repite mientras se
mantenga, para no saturar).

**Limpieza automática:** si Firebase reporta un token como inválido/no registrado (app
desinstalada, token rotado), el microservicio lo desactiva solo — no hace falta limpiarlos a mano.

---

## 9. Requisitos de CPU / RAM

Medido empíricamente (no es una estimación teórica): un solo proceso worker, con **todos** los
modelos de ML cargados (IsolationForest, RandomForest, Algoritmo Genético) y sirviendo
peticiones reales a `/anomalies/detect`, usa:

| Momento | RAM (RSS) |
|---|---|
| Proceso Python recién iniciado, antes de importar la app | ~8 MB |
| Después de cargar FastAPI + todos los modelos de ML | ~190 MB |
| Después de 50 peticiones seguidas a `/detect` | ~192 MB (prácticamente sin crecimiento) |

El clasificador de texto (TF-IDF + Naive Bayes) y BM25 no se cargan al iniciar el servicio (son
"perezosos": solo tocan disco cuando se usan por primera vez) y su artefacto pesa unos KB, así
que no cambian este número de forma relevante.

**CPU:** este servicio es liviano en cómputo — cada predicción es sobre modelos chicos (KB, no
GB) y toma milisegundos; no hay entrenamiento en el proceso web, y BM25/TF-IDF trabajan sobre
corpus de texto corto (decenas de mensajes, no miles). El cuello de botella real, si lo hay, será
de red/BD (esperar a Postgres), no de CPU.

**Recomendación práctica para un piloto/producción chica** (unos cuantos usuarios, ESP32 y
consultas concurrentes moderadas):

| Escenario | vCPU | RAM |
|---|---|---|
| Mínimo viable (1 worker de Uvicorn) | 1 | 512 MB |
| Recomendado (headroom + 2 workers de Uvicorn + poller + picos de tráfico) | 2 | 1 GB |
| Con margen amplio para crecer sin re-dimensionar pronto | 2 | 2 GB |

Esto alcanza cómodamente para: los 3 modelos de ML cargados por worker (~190 MB × N workers),
el pool de conexiones a Postgres, la tarea de fondo del poller, y el sistema operativo. Si en el
futuro se reentrenan `rf_calidad`/`rf_tiempo_restante` con datasets reales mucho más grandes, o
se sube el volumen de sensores/usuarios en varios órdenes de magnitud, vale la pena volver a
medir — pero para el tamaño actual del proyecto, 1 vCPU / 512 MB ya corre el servicio completo
sin problema, y 2 vCPU / 1 GB da margen cómodo de sobra.

---

## 10. Variables de entorno (`.env`)

| Variable | Default | Para qué |
|---|---|---|
| `DATABASE_URL` | sqlite local | Cadena de conexión a Neon en producción |
| `EMAIL_ENABLED` | `false` | Activa notificación por correo (riesgo/crítico) |
| `SMTP_HOST/PORT/USER/PASSWORD/FROM` | — | Credenciales SMTP si `EMAIL_ENABLED=true` |
| `ALERT_EMAIL_TO` | — | Destinatarios separados por coma |
| `EMAIL_MIN_SEVERIDAD` | `riesgo` | Severidad mínima que dispara correo |
| `FCM_ENABLED` | `false` | Activa push (ver sección 8) |
| `FCM_CREDENTIALS_PATH` | — | Ruta al JSON del service account de Firebase |
| `FCM_MIN_SEVERIDAD` | `advertencia` | Severidad mínima que dispara push |
| `POLLING_ENABLED` | `true` | Prende/apaga el poller de tiempo real |
| `POLLING_INTERVALO_SEGUNDOS` | `30` | Cada cuánto revisa el poller |
| `POLLING_BATCH_SIZE` | `50` | Filas por corrida del poller |
| `INTERNAL_API_KEY` | vacío | Header `X-Internal-Api-Key` exigido entre servicios (vacío = sin exigencia, solo dev) |
| `MODELO_VERSION` | `2.0.0` | Se refleja en las respuestas del pipeline |

---

## 11. Pendientes conocidos

- Correr `migration.sql` contra Neon si aún no se aplicó (crea `dispositivos_usuario`,
  `ml_estado_polling`, `reportes_lote`, columnas de riesgo de lluvia en `predicciones`).
- Configurar credenciales reales de Firebase (sección 8) para que el push deje de ser un no-op.
- `rf_calidad.joblib`/`rf_tiempo_restante.joblib` (sección 3.3) siguen sin existir hasta
  acumular suficientes lotes reales terminados.
- El riesgo de lluvia (sección 3.4) puede sobreestimar por el desajuste sensor crudo vs.
  "lluvia sostenida" de entrenamiento.
