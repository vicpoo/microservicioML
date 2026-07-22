-- Migración para microservicioMLL — actualizada tras alinear el código con el
-- esquema REAL de Neon (confirmado contra un pg_dump reciente de la BD en vivo).
--
-- La versión anterior de este archivo asumía columnas (humedad, velocidad_viento,
-- radiacion_solar, lluvia como float normalizado) que NO coinciden con la tabla real:
-- lecturas_ambientales YA tiene presion_hpa, altitud_m, lluvia_analog y
-- lluvia_detectada, y NUNCA tuvo columna de humedad ambiental (el hardware real es
-- BMP280, no BME280). Esta versión ya no toca lecturas_ambientales -no hace falta-
-- y se enfoca en lo que sí falta: la tabla retroalimentacion_ml (con el esquema
-- correcto desde el inicio) y una corrección a inferencias_ml.
--
-- Seguro de correr varias veces (usa IF NOT EXISTS / ON CONFLICT).
-- Ejecutar en Neon (psql, Neon SQL editor, o `psql "$DATABASE_URL" -f migration.sql`).

BEGIN;

-- 1. Índices para acelerar consultas de historial por lote + tiempo (usadas por
--    /api/v1/anomalies). Es probable que ya existan (el dump real ya los trae);
--    IF NOT EXISTS los deja sin efecto si es así.
CREATE INDEX IF NOT EXISTS ix_lecturas_ambientales_lote_ts
    ON public.lecturas_ambientales (id_lote, "timestamp" DESC);

CREATE INDEX IF NOT EXISTS ix_alertas_lote_fecha
    ON public.alertas (id_lote, fecha_generada DESC);

CREATE INDEX IF NOT EXISTS ix_predicciones_lote_fecha
    ON public.predicciones (id_lote, fecha_prediccion DESC);

CREATE INDEX IF NOT EXISTS ix_recomendaciones_lote_fecha
    ON public.recomendaciones (id_lote, fecha_generada DESC);

-- 2. Tabla de retroalimentación real (RNF-19): etiquetas reales que reporta el
--    productor al finalizar un lote (calidad_real + tiempo_real_horas). Tabla NUEVA
--    (no existe todavía en Neon), definida ya alineada con el hardware real: sin
--    humedad_ambiental, con lluvia_detectada booleano en vez de un float sintético.
--    scripts/recolectar_datos_reales.py la usa directamente (ya no hace falta
--    exportarla a CSV aparte).
CREATE TABLE IF NOT EXISTS public.retroalimentacion_ml (
    id_retroalimentacion serial PRIMARY KEY,
    id_lote integer NOT NULL,
    tipo_proceso varchar(50) NOT NULL,
    temperatura_grano numeric(5,2),
    temperatura_ambiental numeric(5,2),
    humedad_grano smallint,
    lluvia_detectada boolean,
    luz numeric(10,2),
    tiempo_real_horas numeric(6,2) NOT NULL,
    calidad_real varchar(20) NOT NULL,
    fecha_reporte timestamp DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_retroalimentacion_ml_lote
    ON public.retroalimentacion_ml (id_lote);

-- 3. inferencias_ml es una tabla LEGADO de un prototipo de clustering anterior a
--    este pipeline; su columna `humedad` quedó NOT NULL de esa época, pero ya no
--    hay humedad ambiental que reportar (BMP280, no BME280). Sin este ALTER, el
--    INSERT que hace app/services/notifier.registrar_inferencia() falla contra la
--    Neon real.
ALTER TABLE public.inferencias_ml
    ALTER COLUMN humedad DROP NOT NULL;

-- 4. Registrar el modelo del pipeline en modelos_ml (idempotente). El microservicio
--    también hace "get-or-create" de esta fila en tiempo de arranque; esto solo la
--    deja pre-sembrada para que predicciones.id_modelo tenga a qué apuntar incluso
--    antes del primer arranque de la app.
INSERT INTO public.modelos_ml (nombre, version, tipo, activo, fecha_entrenamiento)
SELECT 'pipeline_anomalias_mll', '2.0.0', 'isolation_forest+random_forest', true, now()
WHERE NOT EXISTS (
    SELECT 1 FROM public.modelos_ml
    WHERE nombre = 'pipeline_anomalias_mll' AND version = '2.0.0'
);

-- 5. Paso 11 (despliegue): tabla de tokens de dispositivo para notificaciones push (FCM).
--    Pieza que faltaba según definicion_problema_kajve.md Sección 5 ("Aislamiento de alertas
--    por usuario"): la cadena id_sensor -> id_lote -> id_usuario ya existe en el esquema, pero
--    no hay dónde guardar el token FCM de cada dispositivo para poder empujarle la notificación
--    a ESE usuario en particular. Un usuario puede tener más de un dispositivo (teléfono +
--    tablet, o reinstaló la app y le tocó un token nuevo) -- por eso es una tabla aparte y no
--    una columna en el usuario. Sin UNIQUE en fcm_token a propósito: si el mismo token físico
--    reaparece (reinstalación), se actualiza el registro existente en vez de duplicarlo (ver
--    app/api/routes/dispositivos.py::registrar_dispositivo, que hace upsert por
--    (id_usuario, fcm_token)).
CREATE TABLE IF NOT EXISTS public.dispositivos_usuario (
    id_dispositivo serial PRIMARY KEY,
    id_usuario integer NOT NULL,
    fcm_token text NOT NULL,
    plataforma varchar(20) NOT NULL DEFAULT 'android',  -- android | ios | web
    activo boolean NOT NULL DEFAULT true,
    fecha_registro timestamp DEFAULT now(),
    fecha_ultima_actualizacion timestamp DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_dispositivos_usuario_token
    ON public.dispositivos_usuario (id_usuario, fcm_token);

CREATE INDEX IF NOT EXISTS ix_dispositivos_usuario_usuario
    ON public.dispositivos_usuario (id_usuario) WHERE activo;

-- 6. Paso 12 (monitoreo): cursor compartido para que el servicio funcione en tiempo real de
--    verdad. app/services/poller.py revisa lecturas_ambientales cada
--    POLLING_INTERVALO_SEGUNDOS por lecturas nuevas y las procesa solo; POST
--    /internal/lecturas/nuevas (el webhook que llama el Gestor) hace lo mismo al instante.
--    Ambos caminos avanzan esta MISMA fila después de procesar una lectura -- así, sin
--    importar cuál de los dos la procesó primero, el otro la salta y no se duplican
--    predicciones/alertas/notificaciones push para la misma lectura. Fila única (id=1),
--    no hay "varios cursores".
CREATE TABLE IF NOT EXISTS public.ml_estado_polling (
    id integer PRIMARY KEY,
    ultima_id_lectura_procesada integer NOT NULL DEFAULT 0,
    actualizado_en timestamp DEFAULT now()
);

INSERT INTO public.ml_estado_polling (id, ultima_id_lectura_procesada)
SELECT 1, 0
WHERE NOT EXISTS (SELECT 1 FROM public.ml_estado_polling WHERE id = 1);

-- 7. Conecta el Algoritmo Genético de predicción de lluvia (paso 7, entrenado desde el paso 8)
--    a la tabla `predicciones`: hasta ahora ga_lluvia.joblib solo se evaluaba offline
--    (notebooks 07-10), nunca se guardaba un resultado en vivo. Columnas nullable -- no rompen
--    filas existentes ni el esquema para quien no use esta predicción.
ALTER TABLE public.predicciones
    ADD COLUMN IF NOT EXISTS riesgo_lluvia_proxima boolean,
    ADD COLUMN IF NOT EXISTS horas_anticipacion_lluvia smallint;

-- 8. NLP paso 4: historial de reportes en lenguaje natural (ver NLP/README.md). Cada llamada a
--    GET /anomalies/{id_lote}/reporte guarda una fila aquí -- mismo criterio que ya siguen
--    `predicciones`/`alertas`/`recomendaciones` (una fila nueva por generación, no se
--    sobrescribe la anterior), para poder ver cómo cambió el reporte de un lote con el tiempo.
CREATE TABLE IF NOT EXISTS public.reportes_lote (
    id_reporte serial PRIMARY KEY,
    id_lote integer NOT NULL,
    reporte_texto text NOT NULL,
    fecha_generado timestamp DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_reportes_lote_lote_fecha
    ON public.reportes_lote (id_lote, fecha_generado DESC);

-- 9. Cooldown de push para anomalías generales (no lluvia, que ya tiene su propio mecanismo vía
--    predicciones.riesgo_lluvia_proxima). Una fila por (id_lote, tipo_anomalia): se actualiza,
--    no se acumula, cada vez que se envía un push de ese tipo para ese lote -- evita ráfagas si
--    la misma anomalía sigue presente en lecturas consecutivas (ej. el poller cada 30s).
CREATE TABLE IF NOT EXISTS public.ml_ultimo_push_anomalia (
    id_lote integer NOT NULL,
    tipo_anomalia varchar(50) NOT NULL,
    fecha_ultimo_push timestamp NOT NULL DEFAULT now(),
    PRIMARY KEY (id_lote, tipo_anomalia)
);

COMMIT;
