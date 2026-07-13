#app/services/notifier.py
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.alertas import Alerta
from app.models.inferencias_ml import InferenciaML
from app.models.modelos_ml import ModeloML
from app.models.predicciones import Prediccion
from app.models.recomendaciones import Recomendacion

settings = get_settings()

_SEVERIDAD_A_NIVEL = {  # mapea nuestra escala de 4 niveles a la enum nivel_severidad de la BD
    "advertencia": "baja",   # enum nivel_severidad: baja < media < alta < critica
    "riesgo": "alta",
    "critico": "critica",
}
SEVERITY_RANK = {"normal": 0, "advertencia": 1, "riesgo": 2, "critico": 3}

_modelo_id_cache: Optional[int] = None


def get_or_create_modelo(db: Session) -> int:
    global _modelo_id_cache
    if _modelo_id_cache is not None:
        return _modelo_id_cache
    nombre = "pipeline_anomalias_mll"
    modelo = (
        db.query(ModeloML)
        .filter(ModeloML.nombre == nombre, ModeloML.version == settings.modelo_version)
        .first()
    )
    if modelo is None:
        modelo = ModeloML(
            nombre=nombre,
            version=settings.modelo_version,
            tipo="isolation_forest+random_forest",
            activo=True,
            fecha_entrenamiento=datetime.now(timezone.utc),
        )
        db.add(modelo)
        db.commit()
        db.refresh(modelo)
    _modelo_id_cache = modelo.id_modelo
    return _modelo_id_cache


def registrar_inferencia(
    db: Session,
    features: Dict[str, float],
    severidad: str,
    mensaje: str,
) -> int:
    registro = InferenciaML(
        temperatura=features["temperatura_ambiental"],
        humedad=features["humedad_ambiental"],
        cluster_id=SEVERITY_RANK.get(severidad, 0),
        cluster_nombre=severidad,
        recomendacion=mensaje,
        confianza=None,
        modelo_version=settings.modelo_version,
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)
    return registro.id_inferencia


def registrar_alerta(db: Session, id_lote: int, id_sensor: Optional[int], tipo_alerta: str, mensaje: str, severidad: str) -> int:
    alerta = Alerta(
        id_lote=id_lote,
        id_sensor=id_sensor,
        tipo_alerta=tipo_alerta,
        mensaje=mensaje,
        nivel_severidad=_SEVERIDAD_A_NIVEL.get(severidad, "baja"),
    )
    db.add(alerta)
    db.commit()
    db.refresh(alerta)
    return alerta.id_alerta


def registrar_recomendaciones(db: Session, id_lote: int, recomendaciones: List[Dict[str, str]]) -> None:
    for rec in recomendaciones:
        db.add(Recomendacion(id_lote=id_lote, texto=rec["texto"], origen="modelo_ml"))
    db.commit()


def registrar_prediccion(
    db: Session, id_lote: int, id_modelo: int, tiempo_estimado_horas: Optional[float],
    calidad_estimada: Optional[str], confianza: Optional[float],
) -> None:
    db.add(Prediccion(
        id_lote=id_lote,
        id_modelo=id_modelo,
        tiempo_estimado_horas=tiempo_estimado_horas,
        calidad_estimada=calidad_estimada,
        confianza=confianza,
    ))
    db.commit()


def _debe_notificar_email(severidad: str) -> bool:
    if not settings.email_enabled:
        return False
    minimo = SEVERITY_RANK.get(settings.email_min_severidad, 2)
    return SEVERITY_RANK.get(severidad, 0) >= minimo


def enviar_email_alerta(id_lote: Optional[int], severidad: str, mensaje: str, recomendaciones: List[Dict[str, str]]) -> bool:
    """Envía correo de notificación si está habilitado y hay credenciales SMTP configuradas.
    No lanza excepción si falla: registra el error y sigue (la detección/alerta en BD ya quedó
    guardada de todas formas)."""
    if not _debe_notificar_email(severidad):
        return False
    destinatarios = settings.alert_email_recipients
    if not (settings.smtp_host and settings.smtp_user and settings.smtp_password and destinatarios):
        return False

    cuerpo = (
        f"Se detectó una anomalía de severidad '{severidad}' en el lote {id_lote}.\n\n"
        f"{mensaje}\n\nRecomendaciones:\n"
        + "\n".join(f"- {r['texto']}" for r in recomendaciones)
    )
    msg = MIMEText(cuerpo, _charset="utf-8")
    msg["Subject"] = f"[microservicioMLL] Anomalía {severidad.upper()} - Lote {id_lote}"
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = ", ".join(destinatarios)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(msg["From"], destinatarios, msg.as_string())
        return True
    except Exception as exc:  # pragma: no cover - dependemos de red/credenciales reales
        print(f"[notifier] No se pudo enviar el correo de alerta: {exc}")
        return False
