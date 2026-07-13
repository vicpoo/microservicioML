-- Migración para microservicioMLL v2
-- Agrega las columnas que faltan en lecturas_ambientales para soportar
-- los sensores IoT reales (BME280, DS18B20, BH1750, FC-37, humedad de suelo/grano)
-- y deja lista la tabla modelos_ml para el nuevo pipeline de ML.
--
-- Seguro de correr varias veces (usa IF NOT EXISTS / ON CONFLICT).
-- Ejecutar en Neon (psql, Neon SQL editor, o `psql "$DATABASE_URL" -f migration.sql`).

BEGIN;

-- 1. temperatura_grano (DS18B20): no existía, solo estaba "temperatura" (ambiental, BME280)
ALTER TABLE public.lecturas_ambientales
    ADD COLUMN IF NOT EXISTS temperatura_grano numeric(5,2);

-- 2. luz (BH1750, en lux, rango típico 0-65535)
ALTER TABLE public.lecturas_ambientales
    ADD COLUMN IF NOT EXISTS luz numeric(10,2);

-- 3. lluvia (FC-37). Se guarda normalizado 0.000 (seco) a 1.000 (lluvia intensa).
--    El gateway/firmware IoT debe normalizar la lectura analógica del FC-37 antes de enviarla.
ALTER TABLE public.lecturas_ambientales
    ADD COLUMN IF NOT EXISTS lluvia numeric(4,3);

COMMENT ON COLUMN public.lecturas_ambientales.temperatura IS 'Temperatura ambiental (BME280), °C';
COMMENT ON COLUMN public.lecturas_ambientales.humedad IS 'Humedad relativa ambiental (BME280), %';
COMMENT ON COLUMN public.lecturas_ambientales.humedad_grano IS 'Humedad del grano (sensor capacitivo de humedad, colocado en la cama de secado), %';
COMMENT ON COLUMN public.lecturas_ambientales.temperatura_grano IS 'Temperatura del grano (DS18B20), °C';
COMMENT ON COLUMN public.lecturas_ambientales.luz IS 'Irradiancia lumínica (BH1750), lux';
COMMENT ON COLUMN public.lecturas_ambientales.lluvia IS 'Intensidad de lluvia normalizada 0-1 (FC-37)';

-- 4. Índice para acelerar consultas de historial por lote + tiempo (usadas por /api/v1/anomalies)
CREATE INDEX IF NOT EXISTS ix_lecturas_ambientales_lote_ts
    ON public.lecturas_ambientales (id_lote, "timestamp" DESC);

CREATE INDEX IF NOT EXISTS ix_alertas_lote_fecha
    ON public.alertas (id_lote, fecha_generada DESC);

CREATE INDEX IF NOT EXISTS ix_predicciones_lote_fecha
    ON public.predicciones (id_lote, fecha_prediccion DESC);

CREATE INDEX IF NOT EXISTS ix_recomendaciones_lote_fecha
    ON public.recomendaciones (id_lote, fecha_generada DESC);

-- 5. Registrar el modelo del nuevo pipeline en modelos_ml (idempotente).
--    El microservicio también hace "get-or-create" de esta fila en tiempo de arranque,
--    esto solo la deja pre-sembrada para que predicciones.id_modelo tenga a qué apuntar
--    incluso antes del primer arranque de la app.
INSERT INTO public.modelos_ml (nombre, version, tipo, activo, fecha_entrenamiento)
SELECT 'pipeline_anomalias_mll', '2.0.0', 'isolation_forest+random_forest', true, now()
WHERE NOT EXISTS (
    SELECT 1 FROM public.modelos_ml
    WHERE nombre = 'pipeline_anomalias_mll' AND version = '2.0.0'
);

COMMIT;
