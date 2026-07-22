#app/models/notificacion_push_anomalia.py
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.sql import func

from app.models.database import Base


class NotificacionPushAnomalia(Base):
    """Cooldown de push por anomalía general (no lluvia, que ya tiene su propio mecanismo vía
    predicciones.riesgo_lluvia_proxima -- ver notifier.ultimo_riesgo_lluvia()). Una sola fila por
    (id_lote, tipo_anomalia): se actualiza, no se acumula, cada vez que se envía un push nuevo de
    ese tipo para ese lote -- evita ráfagas si la misma anomalía sigue presente en lecturas
    consecutivas (ej. el poller cada 30s durante horas con temperatura_alta sostenida)."""

    __tablename__ = "ml_ultimo_push_anomalia"
    __table_args__ = {"extend_existing": True}

    id_lote = Column(Integer, primary_key=True)
    tipo_anomalia = Column(String(50), primary_key=True)
    fecha_ultimo_push = Column(DateTime, nullable=False, server_default=func.now())
