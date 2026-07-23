#app/models/retroalimentacion_ml.py
from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, SmallInteger, String
from sqlalchemy.sql import func

from app.models.database import Base


class RetroalimentacionML(Base):
    """Resultado real de un lote, reportado por el productor/Gestor al finalizar el secado (RNF-19).

    Tabla NUEVA (aún no existe en Neon; la crea migration.sql) — a diferencia de
    lecturas_ambientales, aquí no hay esquema heredado que respetar, así que ya se
    define directamente alineada con el hardware real: sin humedad_ambiental (BMP280
    no la mide) y con lluvia como booleano (lluvia_detectada), no como float
    sintético. humedad_grano se guarda CRUDO (mismo criterio que
    lecturas_ambientales) para no mezclar unidades calibradas y sin calibrar en la
    misma columna; la conversión a % se hace en tiempo de entrenamiento con
    rules.humedad_grano_raw_a_porcentaje, igual que en el resto del pipeline.

    Cada fila es un ejemplo etiquetado: tipo_proceso + lecturas del final del ciclo +
    calidad_real y tiempo_real_horas verificados en campo (no simulados).
    scripts/recolectar_datos_reales.py la combina con lecturas_ambientales al armar
    el dataset de entrenamiento.

    calidad_real ahora es un puntaje SCA (0-100), no una categoría, y se llena en dos momentos
    distintos (ver migration.sql paso 10): tiempo_real_horas llega al finalizar el lote (INSERT);
    calidad_real llega después, cuando existe un resultado de catación real (UPDATE sobre la
    misma fila). Por eso es nullable y por eso id_lote tiene un índice UNIQUE (una sola fila de
    retroalimentación por lote, para poder hacer upsert).
    """

    __tablename__ = "retroalimentacion_ml"
    __table_args__ = {"extend_existing": True}

    id_retroalimentacion = Column(Integer, primary_key=True, index=True)
    id_lote = Column(Integer, nullable=False, unique=True, index=True)
    tipo_proceso = Column(String(50), nullable=False)
    temperatura_grano = Column(Numeric(5, 2), nullable=True)
    temperatura_ambiental = Column(Numeric(5, 2), nullable=True)
    humedad_grano = Column(SmallInteger, nullable=True)  # crudo de ADC, igual que lecturas_ambientales
    lluvia_detectada = Column(Boolean, nullable=True)
    luz = Column(Numeric(10, 2), nullable=True)
    tiempo_real_horas = Column(Numeric(6, 2), nullable=False)
    calidad_real = Column(Numeric(5, 2), nullable=True)  # puntaje SCA 0-100; null hasta que exista catación
    fecha_reporte = Column(DateTime, server_default=func.now())
