#app/models/lecturas_ambientales.py
from sqlalchemy import Column, DateTime, Integer, Numeric
from sqlalchemy.sql import func

from app.models.database import Base


class LecturaAmbiental(Base):
    __tablename__ = "lecturas_ambientales"
    __table_args__ = {"extend_existing": True}

    id_lectura = Column(Integer, primary_key=True, index=True)
    id_sensor = Column(Integer, nullable=False)
    id_lote = Column(Integer, nullable=False, index=True)
    temperatura = Column(Numeric(5, 2), nullable=True)          # BME280 - ambiental
    humedad = Column(Numeric(5, 2), nullable=True)               # BME280 - ambiental
    timestamp = Column(DateTime, server_default=func.now())
    velocidad_viento = Column(Numeric(5, 2), nullable=True)      # legado, ya no se usa (sin anemómetro)
    radiacion_solar = Column(Numeric(6, 2), nullable=True)       # legado
    humedad_grano = Column(Numeric(5, 2), nullable=True)         # sensor capacitivo de humedad de grano
    temperatura_grano = Column(Numeric(5, 2), nullable=True)     # DS18B20 (columna nueva, ver migration.sql)
    luz = Column(Numeric(10, 2), nullable=True)                  # BH1750 (columna nueva)
    lluvia = Column(Numeric(4, 3), nullable=True)                # FC-37, normalizado 0-1 (columna nueva)
