#app/models/recomendaciones.py
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from app.models.database import Base


class Recomendacion(Base):
    __tablename__ = "recomendaciones"
    __table_args__ = {"extend_existing": True}

    id_recomendacion = Column(Integer, primary_key=True, index=True)
    id_lote = Column(Integer, nullable=False, index=True)
    texto = Column(Text, nullable=False)
    origen = Column(String(50), default="modelo_ml")
    fecha_generada = Column(DateTime, server_default=func.now())
