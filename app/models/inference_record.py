#app/models/inference_record.py
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func
from app.models.database import Base


class InferenceRecord(Base):
    __tablename__ = "inferencias_anomalias"

    id = Column(Integer, primary_key=True, index=True)
    id_lote = Column(Integer, nullable=True)
    tipo_proceso = Column(String(50), nullable=True)
    payload_entrada = Column(Text, nullable=False)
    es_anomalia = Column(Boolean, nullable=False)
    score_anomalia = Column(Float, nullable=False)
    nivel_severidad = Column(String(50), nullable=False)
    mensaje = Column(String(255), nullable=True)
    modelo_version = Column(String(50), nullable=True)
    fecha_inferencia = Column(DateTime(timezone=True), server_default=func.now())
