#app/models/estado_polling.py
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.sql import func

from app.models.database import Base


class EstadoPolling(Base):
    """Cursor compartido entre app/services/poller.py (revisión periódica) y
    POST /internal/lecturas/nuevas (webhook del Gestor) -- ver migration.sql sección 6 para
    el porqué: evita procesar dos veces la misma lectura sin importar cuál de los dos
    caminos la vio primero. Fila única (id=1), no hay "varios cursores" por servicio.
    """

    __tablename__ = "ml_estado_polling"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    ultima_id_lectura_procesada = Column(Integer, nullable=False, default=0)
    actualizado_en = Column(DateTime, server_default=func.now())
