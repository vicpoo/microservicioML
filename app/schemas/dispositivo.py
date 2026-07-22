#app/schemas/dispositivo.py
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RegistrarDispositivoRequest(BaseModel):
    """Lo manda quien hace de puente entre la app móvil y este servicio (típicamente tu API
    principal / Servicio Gestor, reenviando el token que Firebase le dio al celular) -- ver nota
    de arquitectura en app/api/routes/dispositivos.py sobre por qué el MLL no habla directo con
    la app móvil."""
    id_usuario: int = Field(description="Usuario dueño del dispositivo")
    fcm_token: str = Field(min_length=10, description="Token FCM que Firebase le asignó al dispositivo")
    plataforma: Literal["android", "ios", "web"] = "android"


class DispositivoResponse(BaseModel):
    id_dispositivo: int
    id_usuario: int
    plataforma: str
    activo: bool
    fecha_registro: Optional[str] = None
    fecha_ultima_actualizacion: Optional[str] = None


class DesactivarDispositivoRequest(BaseModel):
    id_usuario: int
    fcm_token: str
