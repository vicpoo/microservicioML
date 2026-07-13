#app/models/modelos_ml.py
from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String

from app.models.database import Base


class ModeloML(Base):
    __tablename__ = "modelos_ml"
    __table_args__ = {"extend_existing": True}

    id_modelo = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    tipo = Column(String(50), nullable=True)
    rmse = Column(Numeric(6, 3), nullable=True)
    activo = Column(Boolean, default=True)
    fecha_entrenamiento = Column(DateTime, nullable=True)
