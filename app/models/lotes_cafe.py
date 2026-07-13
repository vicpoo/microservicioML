#app/models/lotes_cafe.py
from sqlalchemy import Column, DateTime, Integer, Numeric, String

from app.models.database import Base


class LoteCafe(Base):
    """Mapeo de solo-lectura de la tabla existente lotes_cafe (dueña: backend principal)."""

    __tablename__ = "lotes_cafe"
    __table_args__ = {"extend_existing": True}

    id_lote = Column(Integer, primary_key=True, index=True)
    id_usuario = Column(Integer, nullable=False)
    id_sensor = Column(Integer, nullable=True)
    nombre_lote = Column(String(100), nullable=False)
    variedad = Column(String(100), nullable=True)
    peso_kg = Column(Numeric(10, 2), nullable=True)
    ubicacion = Column(String(200), nullable=True)
    codigo_qr = Column(String(100), nullable=False)
    estado = Column(String(50), nullable=True)
    fecha_inicio_secado = Column(DateTime, nullable=True)
    fecha_fin_secado = Column(DateTime, nullable=True)
    tipo_proceso = Column(String(50), nullable=True)
