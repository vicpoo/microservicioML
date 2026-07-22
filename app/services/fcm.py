#app/services/fcm.py
"""
Envío de notificaciones push vía Firebase Cloud Messaging (FCM). Paso 11 del pipeline de ML
(despliegue): el modelo entrenado ya escribe en predicciones/alertas/recomendaciones (ver
app/services/notifier.py); este módulo es la pieza que faltaba para "disparar la notificación
push" mencionada en la definición del problema (Sección 5, aislamiento de alertas por usuario).

Requiere:
  - `pip install firebase-admin` (ya en requirements.txt).
  - Un service account JSON de un proyecto Firebase (Firebase Console > Configuración del
    proyecto > Cuentas de servicio > Generar nueva clave privada). NO se sube al repo.
  - FCM_CREDENTIALS_PATH apuntando a ese archivo y FCM_ENABLED=true en .env.

Si FCM_ENABLED=false (default) o faltan credenciales, `enviar_push()` no hace nada y no lanza
excepción -- mismo patrón que `notifier.enviar_email_alerta()`: la alerta ya quedó guardada en la
BD de todas formas, un fallo de notificación nunca debe tumbar la petición.
"""
import logging
from typing import Dict, Iterable, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_firebase_app = None
_intento_inicializacion = False


def _obtener_app():
    """Inicializa el SDK de firebase-admin una sola vez por proceso (perezoso: solo si
    FCM_ENABLED=true). Devuelve None si no está configurado o si falla la inicialización --
    nunca lanza, y no reintenta en cada alerta si ya falló una vez."""
    global _firebase_app, _intento_inicializacion
    settings = get_settings()
    if not settings.fcm_enabled:
        return None
    if _firebase_app is not None:
        return _firebase_app
    if _intento_inicializacion:
        return None
    _intento_inicializacion = True
    try:
        import firebase_admin
        from firebase_admin import credentials

        if not settings.fcm_credentials_path:
            logger.warning("[fcm] FCM_ENABLED=true pero falta FCM_CREDENTIALS_PATH; push deshabilitado.")
            return None
        cred = credentials.Certificate(settings.fcm_credentials_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        return _firebase_app
    except Exception as exc:  # pragma: no cover - depende de credenciales/red reales
        logger.warning(f"[fcm] No se pudo inicializar firebase-admin: {exc}")
        return None


def reset_para_pruebas() -> None:
    """Solo para tests: limpia el estado de inicialización cacheado entre pruebas."""
    global _firebase_app, _intento_inicializacion
    _firebase_app = None
    _intento_inicializacion = False


def enviar_push(tokens: Iterable[str], titulo: str, cuerpo: str, datos: Optional[Dict] = None) -> dict:
    """Envía la misma notificación a una lista de tokens FCM (multicast).

    Devuelve {"enviados": int, "fallidos": int, "tokens_invalidos": [...]}. Nunca lanza
    excepción: si FCM no está configurado o falla, regresa enviados=0 en silencio (quien llama
    ya guardó la alerta en la BD independientemente de esto, ver notifier.enviar_push_alerta)."""
    tokens = [t for t in tokens if t]
    resultado = {"enviados": 0, "fallidos": 0, "tokens_invalidos": []}
    if not tokens:
        return resultado

    app = _obtener_app()
    if app is None:
        return resultado  # FCM deshabilitado o mal configurado: no-op silencioso

    try:
        from firebase_admin import messaging

        mensaje = messaging.MulticastMessage(
            notification=messaging.Notification(title=titulo, body=cuerpo),
            data={k: str(v) for k, v in (datos or {}).items()},
            tokens=list(tokens),
        )
        respuesta = messaging.send_each_for_multicast(mensaje, app=app)
        resultado["enviados"] = respuesta.success_count
        resultado["fallidos"] = respuesta.failure_count
        for token, resp in zip(tokens, respuesta.responses):
            if not resp.success and resp.exception is not None:
                codigo = getattr(resp.exception, "code", "")
                if codigo in ("UNREGISTERED", "INVALID_ARGUMENT"):
                    resultado["tokens_invalidos"].append(token)
    except Exception as exc:  # pragma: no cover - depende de red/credenciales reales
        logger.warning(f"[fcm] Error enviando notificación push: {exc}")
        resultado["fallidos"] = len(tokens)

    return resultado
