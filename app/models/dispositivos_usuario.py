#app/models/dispositivos_usuario.py
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from app.models.database import Base


class DispositivoUsuario(Base):
    """Token de dispositivo (FCM) de un usuario, para notificaciones push (paso 11: despliegue).

    Un usuario puede tener varios dispositivos activos (teléfono + tablet, o un token nuevo tras
    reinstalar la app) -- por eso es tabla aparte y no una columna en el usuario. El registro se
    identifica por (id_usuario, fcm_token): ver app/api/routes/dispositivos.py, que hace upsert
    en vez de insertar duplicados cada vez que la app reenvía el mismo token.
    """

    __tablename__ = "dispositivos_usuario"
    __table_args__ = {"extend_existing": True}

    id_dispositivo = Column(Integer, primary_key=True, index=True)
    id_usuario = Column(Integer, nullable=False, index=True)
    fcm_token = Column(Text, nullable=False)
    plataforma = Column(String(20), nullable=False, default="android")  # android | ios | web
    activo = Column(Boolean, nullable=False, default=True)
    fecha_registro = Column(DateTime, server_default=func.now())
    fecha_ultima_actualizacion = Column(DateTime, server_default=func.now(), onupdate=func.now())
