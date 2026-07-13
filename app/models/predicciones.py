#app/models/predicciones.py
from sqlalchemy import Column, DateTime, Integer, Numeric, String
from sqlalchemy.sql import func

from app.models.database import Base


class Prediccion(Base):
    __tablename__ = "predicciones"
    __table_args__ = {"extend_existing": True}

    id_prediccion = Column(Integer, primary_key=True, index=True)
    id_lote = Column(Integer, nullable=False, index=True)
    id_modelo = Column(Integer, nullable=False)
    tiempo_estimado_horas = Column(Numeric(5, 2), nullable=True)
    calidad_estimada = Column(String(50), nullable=True)
    confianza = Column(Numeric(5, 2), nullable=True)
    fecha_prediccion = Column(DateTime, server_default=func.now())
