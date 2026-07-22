#app/models/inferencias_ml.py
from sqlalchemy import Column, DateTime, Integer, Numeric, String, Text
from sqlalchemy.sql import func

from app.models.database import Base


class InferenciaML(Base):
    """Bitácora plana de TODAS las inferencias (con o sin id_lote).

    Reutiliza la tabla inferencias_ml ya existente en el esquema (originalmente
    pensada para un modelo de clustering). Mapeo:
      - temperatura / humedad     -> lecturas ambientales usadas en la inferencia
      - cluster_id / cluster_nombre -> código y nombre del nivel de severidad detectado
      - recomendacion             -> mensaje principal generado
      - confianza                 -> score de confianza del modelo (0-100)
    """

    __tablename__ = "inferencias_ml"
    __table_args__ = {"extend_existing": True}

    id_inferencia = Column(Integer, primary_key=True, index=True)
    temperatura = Column(Numeric(5, 2), nullable=False)
    # NOT NULL en el esquema original (prototipo de clustering, pre-BMP280); se manda
    # None desde notifier.py porque humedad_ambiental ya no existe como variable.
    # Requiere migration.sql: ALTER COLUMN humedad DROP NOT NULL (ver Sección 7 de la
    # migración actualizada) antes de que el INSERT funcione contra la Neon real.
    humedad = Column(Numeric(5, 2), nullable=True)
    cluster_id = Column(Integer, nullable=False)
    cluster_nombre = Column(String(50), nullable=False)
    recomendacion = Column(Text, nullable=False)
    confianza = Column(Numeric(5, 2), nullable=True)
    modelo_version = Column(String(20), nullable=False)
    fecha_inferencia = Column(DateTime, server_default=func.now())
