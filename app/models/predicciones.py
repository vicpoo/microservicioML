#app/models/predicciones.py
from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, SmallInteger
from sqlalchemy.sql import func

from app.models.database import Base


class Prediccion(Base):
    __tablename__ = "predicciones"
    __table_args__ = {"extend_existing": True}

    id_prediccion = Column(Integer, primary_key=True, index=True)
    id_lote = Column(Integer, nullable=False, index=True)
    id_modelo = Column(Integer, nullable=False)
    tiempo_estimado_horas = Column(Numeric(5, 2), nullable=True)
    # Puntaje 0-100 (escala SCA), no una categoría -- es una aproximación del ML durante el
    # secado a partir de sensores, no una catación real (ver Documento de Calidad del Café,
    # Sección 7, y migration.sql paso 10 para el detalle de la migración).
    calidad_estimada = Column(Numeric(5, 2), nullable=True)
    confianza = Column(Numeric(5, 2), nullable=True)
    # Salida del Algoritmo Genético (paso 7/11, app/services/rain_predictor.py) -- riesgo de
    # que llueva en las próximas horas_anticipacion_lluvia horas, no si está lloviendo ahora
    # (eso ya lo cubre el sensor FC-37 vía las reglas/alertas normales).
    riesgo_lluvia_proxima = Column(Boolean, nullable=True)
    horas_anticipacion_lluvia = Column(SmallInteger, nullable=True)
    fecha_prediccion = Column(DateTime, server_default=func.now())
