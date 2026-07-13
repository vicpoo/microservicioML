#app/models/alertas.py
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from app.models.database import Base


class Alerta(Base):
    __tablename__ = "alertas"
    __table_args__ = {"extend_existing": True}

    id_alerta = Column(Integer, primary_key=True, index=True)
    id_lote = Column(Integer, nullable=False, index=True)
    id_sensor = Column(Integer, nullable=True)
    tipo_alerta = Column(String(100), nullable=False)
    mensaje = Column(Text, nullable=True)
    nivel_severidad = Column(String(20), nullable=False)  # enum en Postgres: baja/media/alta/critica
    atendida = Column(Boolean, default=False)
    fecha_generada = Column(DateTime, server_default=func.now())
    fecha_atencion = Column(DateTime, nullable=True)
