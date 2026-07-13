# Archivo: app/core/config.py
# Carpeta: microservicioMLL/app/core/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "microservicioMLL"

    # --- Base de datos ---
    # Producción: cadena de conexión de Neon (postgresql+psycopg2://user:pass@host/db?sslmode=require&channel_binding=require)
    # Desarrollo/tests: si no se define, cae a sqlite local para poder correr sin red.
    database_url: str = "sqlite:///./app.db"

    # --- Correo (notificaciones de anomalías riesgo/crítico) ---
    email_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None
    # Lista separada por comas de destinatarios, ej: "productor@correo.com,admin@correo.com"
    alert_email_to: Optional[str] = None
    # Severidad mínima que dispara correo: advertencia | riesgo | critico
    email_min_severidad: str = "riesgo"

    # --- Seguridad entre servicios ---
    # El MLL es un servicio interno: solo lo llaman el Servicio Gestor (para avisarle de
    # lecturas nuevas) y, si tu API móvil decide consultarlo en vez de leer Neon directo,
    # también ella. Todos deben mandar este mismo valor en el header X-Internal-Api-Key.
    # Vacío = sin exigencia (solo para desarrollo local).
    internal_api_key: Optional[str] = None

    # --- Modelo ---
    modelo_version: str = "2.0.0"

    class Config:
        env_file = ".env"

    @property
    def alert_email_recipients(self) -> List[str]:
        if not self.alert_email_to:
            return []
        return [addr.strip() for addr in self.alert_email_to.split(",") if addr.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()