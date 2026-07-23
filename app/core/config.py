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

    # --- Notificaciones push (FCM, paso 11: despliegue) ---
    # Requiere un service account JSON de un proyecto Firebase (Firebase Console >
    # Configuración del proyecto > Cuentas de servicio > Generar nueva clave privada).
    # Si fcm_enabled=False (default) o falta la ruta de credenciales, notifier.enviar_push_alerta
    # no hace nada y no lanza excepción -- mismo patrón que el envío de correo.
    fcm_enabled: bool = False
    fcm_credentials_path: Optional[str] = None
    # Severidad mínima que dispara push: advertencia | riesgo | critico. A diferencia del
    # correo (que solo avisa en riesgo/critico para no saturar), el push por defecto dispara
    # desde "advertencia" -- o sea, en CUALQUIER anomalía que el ML detecte, no solo las
    # graves (ver app/api/routes/inference.py::ejecutar_pipeline, el push ya no depende de
    # que se haya generado una alerta formal).
    fcm_min_severidad: str = "advertencia"
    # Minutos mínimos entre dos push de anomalía para el mismo lote+tipo, evita ráfagas si la
    # condición persiste en lecturas consecutivas (ej. el poller cada 30s durante horas con la
    # misma anomalía sostenida). No aplica al push de riesgo de lluvia, que ya tiene su propio
    # debounce por transición (ver notifier.ultimo_riesgo_lluvia()).
    fcm_cooldown_minutos: int = 30

    # --- Polling en tiempo real (paso 12: monitoreo) ---
    # El disparador normal es que el Gestor llame POST /internal/lecturas/nuevas apenas
    # inserta una lectura. Este poller es una RED DE SEGURIDAD: revisa lecturas_ambientales
    # cada polling_intervalo_segundos por si algo se coló sin avisar (Gestor caído, bug de
    # red, etc.), así el servicio sigue siendo "tiempo real" aunque el webhook falle. Usa un
    # cursor compartido (tabla ml_estado_polling) con el webhook para no procesar la misma
    # lectura dos veces -- ver app/services/poller.py.
    polling_enabled: bool = True
    polling_intervalo_segundos: int = 30
    polling_batch_size: int = 50

    # --- Reentrenamiento automático (cierra el ciclo "el modelo aprende de sus errores": si
    # predijo 70 y la realidad fue 90, esto hace que la PRÓXIMA predicción ya lo tenga en
    # cuenta) ---
    # Apagado por default a propósito: entrenar consume CPU por varios segundos/minutos y, aunque
    # corre en un hilo aparte (no bloquea el event loop), sigue compitiendo por CPU con las
    # peticiones en vivo del mismo proceso -- ver app/services/reentrenador.py.
    reentrenamiento_automatico_enabled: bool = False
    reentrenamiento_intervalo_horas: int = 24

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