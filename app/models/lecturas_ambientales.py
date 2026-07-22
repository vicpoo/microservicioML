#app/models/lecturas_ambientales.py
from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, SmallInteger
from sqlalchemy.sql import func

from app.models.database import Base


class LecturaAmbiental(Base):
    """Mapeo EXACTO de la tabla real en Neon (confirmado contra
    respaldo-cafe-datos-sensores-v2.sql, pg_dump del esquema en vivo).

    El hardware real usa BMP280 (no BME280): no hay columna de humedad
    ambiental. Tampoco hay velocidad_viento/radiacion_solar (nunca se
    instaló anemómetro). "lluvia" no es un float normalizado: el firmware
    manda un booleano ya resuelto (lluvia_detectada) más el valor crudo del
    ADC (lluvia_analog, sin calibrar, ver definicion_problema_kajve.md
    Sección 6). humedad_grano también es crudo (smallint del ADC), no un
    porcentaje calibrado.
    """

    __tablename__ = "lecturas_ambientales"
    __table_args__ = {"extend_existing": True}

    id_lectura = Column(Integer, primary_key=True, index=True)
    id_sensor = Column(Integer, nullable=False)
    id_lote = Column(Integer, nullable=False, index=True)
    temperatura = Column(Numeric(5, 2), nullable=True)            # BMP280 - ambiental
    timestamp = Column(DateTime, server_default=func.now())
    humedad_grano = Column(SmallInteger, nullable=True)           # sensor capacitivo, valor CRUDO del ADC (sin calibrar)
    temperatura_grano = Column(Numeric(5, 2), nullable=True)      # DS18B20
    luz = Column(Numeric(10, 2), nullable=True)                   # BH1750, lux
    presion_hpa = Column(Numeric(7, 3), nullable=True)            # BMP280
    altitud_m = Column(Numeric(8, 3), nullable=True)              # BMP280 (derivada de la presión)
    lluvia_analog = Column(SmallInteger, nullable=True)           # FC-37, valor CRUDO del ADC (sin calibrar)
    lluvia_detectada = Column(Boolean, nullable=True)             # FC-37, booleano ya resuelto por el firmware
