# microservicioMLL — kajve

Microservicio FastAPI de Machine Learning para **kajve**: monitoreo del secado de café con
sensores ESP32. Recibe lecturas ambientales (temperatura de grano/ambiente, humedad de grano,
lluvia, luz), detecta anomalías, predice tiempo restante y calidad estimada, genera
recomendaciones, y dispara alertas (correo + push) cuando la severidad lo amerita.

No usa dataset sintético: entrena con datos reales de sensores ESP32 en campo, recolectados de
la base de datos real (Neon/Postgres) que comparte con el backend principal ("Gestor").

## 1. Arquitectura

```
Sensores ESP32 --> Gestor (backend principal) --> Neon (Postgres, BD compartida)
                                                        |
                                    POST /internal/lecturas/nuevas (webhook, instantáneo)
                                    ó detectado solo por el poller cada N segundos (red de
                                    seguridad si el webhook no llegó a llamarse)
                                                        v
                                          microservicioMLL (este proyecto)
                                     Reglas de dominio + IsolationForest +
                                     RandomForest + Algoritmo Genético (lluvia)
                                                        |
                        predicciones / alertas / recomendaciones (Neon)
                                                        |
                          correo (SMTP, riesgo/critico) + push FCM (cualquier anomalía),
                          siempre solo al dueño real del lote (id_lote -> lotes_cafe.id_usuario
                          -> dispositivos_usuario), nunca a otros usuarios
```

Motor de decisión por request (`app/api/routes/inference.py::ejecutar_pipeline`):

1. **Reglas de dominio** (`app/services/rules.py`) — umbrales fijos, fuente autoritativa para
   severidad cuando detectan algo.
2. **RandomForestClassifier** (tipo de anomalía) + **IsolationForest** (outliers no vistos por
   las reglas) — pueden elevar la severidad, nunca bajarla.
3. **Algoritmo Genético** (`app/services/rain_predictor.py`, riesgo de lluvia en las próximas
   horas — no si está lloviendo ahora, eso ya lo cubre el sensor) y **RandomForest** (tiempo
   restante/calidad, pendientes de datos reales suficientes) — ver `ML/README.md`.
4. **Recomendaciones** deterministas por tipo de anomalía (`app/services/recommender.py`).
5. Persistencia en `predicciones` / `alertas` / `recomendaciones`, correo si la severidad es
   `riesgo`/`critico`, push (FCM) en **cualquier anomalía detectada** (`es_anomalia=True`,
   umbral configurable con `FCM_MIN_SEVERIDAD`), y push adicional (con debounce) cuando el
   riesgo de lluvia pasa a `True` — siempre y solo al dueño real del lote.

### Aislamiento por usuario (multi-tenant)

Cada ESP32 está asignado a un lote (`lotes_cafe.id_sensor`), y cada lote tiene un dueño
(`lotes_cafe.id_usuario`). TODO lo que el ML escribe o notifica pasa por esa cadena antes de
tocar datos de un usuario:

- **Push (FCM):** `notifier.enviar_push_alerta` resuelve `id_lote -> lotes_cafe.id_usuario ->
  dispositivos_usuario` y solo manda el token del dueño real — nunca a todos los dispositivos
  registrados. Probado explícitamente: dos usuarios con lotes distintos, y el push de uno
  jamás llega al token del otro.
- **Lectura (`GET /anomalies*`):** cada endpoint exige `id_usuario` como query param y llama
  `_verificar_dueno()` (`history.py`) antes de devolver nada — 403 si el lote no es tuyo.
- **Escritura (`POST /anomalies/detect`):** 403 si el `id_usuario` del body no coincide con el
  dueño real del lote en `lotes_cafe`.

RLS (row level security) de Postgres no cubre este flujo por sí solo — el filtrado es
explícito en código en cada uno de estos tres puntos.

### Tiempo real: webhook + poller

El disparador preferido sigue siendo que el Gestor llame `POST /internal/lecturas/nuevas` al
instante. Pero para que el servicio sea de verdad "tiempo real" y no dependa 100% de que ese
aviso llegue, `app/services/poller.py` corre como tarea de fondo dentro del mismo proceso y
revisa `lecturas_ambientales` cada `POLLING_INTERVALO_SEGUNDOS` (default 30s) por lecturas que
nadie avisó. Webhook y poller comparten un cursor (tabla `ml_estado_polling`) para no procesar
la misma lectura dos veces sin importar cuál de los dos la vio primero.

## 2. El pipeline de ML completo

Todo el trabajo de datos (recolección → limpieza → features → modelos → evaluación → ajuste →
despliegue → monitoreo) vive documentado paso a paso en **[`ML/README.md`](ML/README.md)**, con
notebooks ejecutables y módulos reutilizables (`ML/*.py`) que reflejan exactamente lo que corre
en producción. Ese documento es la fuente de verdad del pipeline de ML — este README solo cubre
el microservicio como API.

## 3. Endpoints

| Método | Ruta | Quién la llama | Qué hace |
|---|---|---|---|
| POST | `/api/v1/anomalies/detect` | manual / pruebas (curl, Postman) | Corre el pipeline completo con 6 lecturas en el body |
| POST | `/api/v1/internal/lecturas/nuevas` | Gestor | El Gestor avisa "hay lectura nueva"; el MLL la lee de Neon y corre el pipeline |
| POST | `/api/v1/internal/lotes/{id_lote}/resultado-real` | Gestor | Registra calidad/tiempo real reportado por el productor al terminar un lote (retroalimentación) |
| GET | `/api/v1/internal/monitoreo/salud` | Gestor / cron | Paso 12: desempeño en producción vs baseline, tasa de alertas, si conviene reentrenar |
| GET | `/api/v1/anomalies` | app móvil / Gestor | Historial de anomalías de un lote |
| GET | `/api/v1/anomalies/{id_lote}/predicciones` | app móvil / Gestor | Historial de predicciones de un lote |
| GET | `/api/v1/anomalies/{id_lote}/recomendaciones` | app móvil / Gestor | Historial de recomendaciones de un lote |
| GET | `/api/v1/anomalies/{id_lote}/reporte` | app móvil / Gestor | Genera (y guarda) el reporte en lenguaje natural del lote (PLN, ver `NLP/README.md`) |
| GET | `/api/v1/anomalies/{id_lote}/reportes` | app móvil / Gestor | Historial de reportes ya generados de un lote |
| GET | `/api/v1/anomalies/reportes/buscar` | app móvil / Gestor | Busca con BM25 en todo el historial de reportes del usuario (todos sus lotes), ver `NLP/README.md` |
| POST | `/api/v1/nlp/clasificar-texto` | app móvil / Gestor | Sugiere severidad (TF-IDF + Naive Bayes) para texto libre en español, ver `NLP/README.md` |
| POST | `/api/v1/dispositivos/registrar` | app móvil | Registra/actualiza el token FCM de un dispositivo |
| POST | `/api/v1/dispositivos/desactivar` | app móvil | Desactiva un token FCM (logout) |
| GET | `/api/v1/dispositivos/{id_usuario}` | diagnóstico | Lista dispositivos activos de un usuario |
| GET | `/health` | monitoreo de infraestructura | Liveness check, sin auth |

Todas las rutas bajo `/api/v1` (excepto las de solo lectura pensadas para la app móvil) están
protegidas con el header `X-Internal-Api-Key` (`app/core/security.py`), configurado en
`INTERNAL_API_KEY`.

## 4. Configuración (`.env`)

Ver `env.example` para la lista completa comentada. Resumen:

- `DATABASE_URL` — Postgres (Neon) en producción; `sqlite:///./app.db` para desarrollo local sin red.
- `EMAIL_ENABLED` / `SMTP_*` / `ALERT_EMAIL_TO` / `EMAIL_MIN_SEVERIDAD` — alertas por correo (default: solo riesgo/critico).
- `FCM_ENABLED` / `FCM_CREDENTIALS_PATH` / `FCM_MIN_SEVERIDAD` — alertas push (default: cualquier anomalía, `advertencia` en adelante).
- `POLLING_ENABLED` / `POLLING_INTERVALO_SEGUNDOS` / `POLLING_BATCH_SIZE` — revisión periódica de lecturas nuevas (red de seguridad del tiempo real).
- `INTERNAL_API_KEY` — clave compartida entre el Gestor y el MLL.
- `MODELO_VERSION` — versión reportada en cada respuesta de inferencia.

## 5. Correr localmente

```bash
pip install -r requirements.txt
cp env.example .env   # y edita los valores que apliquen
uvicorn app.main:app --reload
```

Tests (usan sqlite en memoria/local, no tocan Neon):

```bash
pytest tests/test_api.py -q
```

Monitoreo manual (ver [`ML/README.md`](ML/README.md), paso 12):

```bash
python scripts/monitorear_modelos.py
```

## 6. Pendientes conocidos

- Correr `migration.sql` contra Neon (crea `retroalimentacion_ml`, `dispositivos_usuario`,
  `ml_estado_polling`, `reportes_lote`, y quita el `NOT NULL` de `inferencias_ml.humedad`) —
  **todavía no aplicado en producción**.
- Configurar credenciales reales de Firebase (`FCM_CREDENTIALS_PATH`) para que las push
  notifications dejen de ser un no-op.
- `rf_calidad.joblib` y `rf_tiempo_restante.joblib` se eliminaron de `app/ml/artifacts/`: eran
  artefactos viejos entrenados con el dataset sintético deprecado (esquema con
  `humedad_ambiental`, que el hardware real no produce) y siempre fallaban en silencio contra
  datos reales. Mientras no existan, `predictor.py` devuelve `tiempo_estimado_horas` y
  `calidad_estimada` en `null` (mismo resultado que antes, sin cargar un artefacto inútil) hasta
  correr `scripts/train_models.py` con suficientes lotes reales finalizados — ver paso 12 en
  `ML/README.md`.
- El riesgo de lluvia (`prediccion.riesgo_lluvia_proxima`) ya corre en vivo, pero se entrenó
  con una señal de lluvia "sostenida" que hoy solo existe offline (paso 4) — en producción usa
  el sensor crudo, lo que puede sobreestimar el riesgo por el parpadeo conocido del FC-37. Ver
  el hallazgo del paso 11 en `ML/README.md`.
- Decidir si se unifica `scripts/train_models.py` (su propia limpieza simplificada) con el
  pipeline de `ML/` (más riguroso) — detalle en `ML/README.md`.
