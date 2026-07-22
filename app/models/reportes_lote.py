#app/models/reportes_lote.py
from sqlalchemy import Column, DateTime, Integer, Text
from sqlalchemy.sql import func

from app.models.database import Base


class ReporteLote(Base):
    """Historial de reportes en lenguaje natural (PLN, ver NLP/README.md paso 4). Una fila por
    cada vez que se generó un reporte -- mismo criterio que `predicciones`/`alertas`/
    `recomendaciones`: se acumula, no se sobrescribe, para poder ver cómo cambió el reporte de
    un lote con el tiempo."""

    __tablename__ = "reportes_lote"
    __table_args__ = {"extend_existing": True}

    id_reporte = Column(Integer, primary_key=True, index=True)
    id_lote = Column(Integer, nullable=False, index=True)
    reporte_texto = Column(Text, nullable=False)
    fecha_generado = Column(DateTime, server_default=func.now())
