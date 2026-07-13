#app/models/sensores.py
from sqlalchemy import Boolean, Column, Integer, String

from app.models.database import Base


class Sensor(Base):
    """Mapeo de solo-lectura de la tabla existente sensores."""

    __tablename__ = "sensores"
    __table_args__ = {"extend_existing": True}

    id_sensor = Column(Integer, primary_key=True, index=True)
    mac_address = Column(String(50), nullable=False)
    tipo = Column(String(50), nullable=False)
    modelo = Column(String(100), nullable=True)
    estado = Column(String(50), nullable=True)
    mide_viento = Column(Boolean, default=False)
    mide_radiacion = Column(Boolean, default=False)
    mide_humedad_grano = Column(Boolean, default=False)
