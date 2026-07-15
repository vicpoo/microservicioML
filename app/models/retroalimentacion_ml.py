#app/models/retroalimentacion_ml.py
from sqlalchemy import Column, DateTime, Integer, Numeric, String
from sqlalchemy.sql import func

from app.models.database import Base


class RetroalimentacionML(Base):
    """Resultado real de un lote, reportado por el productor/Gestor al finalizar el secado (RNF-19).

    Cada fila es un ejemplo etiquetado con el mismo esquema de columnas que el dataset
    sintético (scripts/generar_dataset.py: tipo_proceso + 6 lecturas + horas_transcurridas +
    calidad_final), pero con calidad_real y tiempo_real_horas verificados en campo en vez de
    simulados. Tabla nueva y separada del dataset sintético a propósito: no se mezclan datos
    simulados con reales en el mismo origen. scripts/exportar_retroalimentacion.py la vuelca a
    CSV y scripts/train_models.py la combina con el dataset sintético al reentrenar.
    """

    __tablename__ = "retroalimentacion_ml"
    __table_args__ = {"extend_existing": True}

    id_retroalimentacion = Column(Integer, primary_key=True, index=True)
    id_lote = Column(Integer, nullable=False, index=True)
    tipo_proceso = Column(String(50), nullable=False)
    temperatura_grano = Column(Numeric(5, 2), nullable=True)
    temperatura_ambiental = Column(Numeric(5, 2), nullable=True)
    humedad_ambiental = Column(Numeric(5, 2), nullable=True)
    humedad_grano = Column(Numeric(5, 2), nullable=True)
    lluvia = Column(Numeric(4, 3), nullable=True)
    luz = Column(Numeric(10, 2), nullable=True)
    tiempo_real_horas = Column(Numeric(6, 2), nullable=False)
    calidad_real = Column(String(20), nullable=False)
    fecha_reporte = Column(DateTime, server_default=func.now())
