#app/services/notifier.py
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.alertas import Alerta
from app.models.dispositivos_usuario import DispositivoUsuario
from app.models.inferencias_ml import InferenciaML
from app.models.lotes_cafe import LoteCafe
from app.models.modelos_ml import ModeloML
from app.models.notificacion_push_anomalia import NotificacionPushAnomalia
from app.models.predicciones import Prediccion
from app.models.recomendaciones import Recomendacion
from app.services import fcm

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
    # OJO: inferencias_ml es una tabla LEGADO de un prototipo de clustering anterior a
    # este pipeline (ver definicion_problema_kajve.md). Su columna `humedad` es
    # NOT NULL en el esquema real, pero ya no existe humedad_ambiental como variable
    # (BMP280, no BME280). Se manda None: hace falta correr la migración que quita
    # ese NOT NULL (ver migration.sql) antes de que este INSERT funcione contra Neon.
    registro = InferenciaML(
        temperatura=features["temperatura_ambiental"],
        humedad=None,
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


def _recomendacion_reciente_duplicada(db: Session, id_lote: int, texto: str) -> bool:
    """Evita filas duplicadas en recomendaciones cuando el mismo problema persiste en lecturas
    consecutivas (ej. el poller cada 30s durante horas) -- mismo patrón que
    debe_notificar_anomalia(): un SELECT fresco contra la tabla real, ANTES de insertar, en vez
    de acumular una fila idéntica por cada ciclo. Reutiliza FCM_COOLDOWN_MINUTOS -- recomendaciones
    no tiene columna de "tipo" separada, así que el duplicado se identifica por texto idéntico
    para el mismo lote, igual que ya se hacía para deduplicar dentro de un mismo request en
    Recommender.generar()."""
    ahora_naive_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    limite = ahora_naive_utc - timedelta(minutes=settings.fcm_cooldown_minutos)
    existente = (
        db.query(Recomendacion)
        .filter(
            Recomendacion.id_lote == id_lote,
            Recomendacion.texto == texto,
            Recomendacion.fecha_generada > limite,
        )
        .first()
    )
    return existente is not None


def registrar_recomendaciones(db: Session, id_lote: int, recomendaciones: List[Dict[str, str]]) -> None:
    for rec in recomendaciones:
        if _recomendacion_reciente_duplicada(db, id_lote, rec["texto"]):
            continue
        db.add(Recomendacion(id_lote=id_lote, texto=rec["texto"], origen="modelo_ml"))
    db.commit()


def registrar_prediccion(
    db: Session, id_lote: int, id_modelo: int, tiempo_estimado_horas: Optional[float],
    calidad_estimada: Optional[float], confianza: Optional[float],
    riesgo_lluvia_proxima: Optional[bool] = None, horas_anticipacion_lluvia: Optional[int] = None,
) -> None:
    db.add(Prediccion(
        id_lote=id_lote,
        id_modelo=id_modelo,
        tiempo_estimado_horas=tiempo_estimado_horas,
        calidad_estimada=calidad_estimada,
        confianza=confianza,
        riesgo_lluvia_proxima=riesgo_lluvia_proxima,
        horas_anticipacion_lluvia=horas_anticipacion_lluvia,
    ))
    db.commit()


def ultimo_riesgo_lluvia(db: Session, id_lote: int) -> Optional[bool]:
    """Último valor conocido de riesgo_lluvia_proxima para este lote (antes de registrar la
    predicción nueva) -- lo usa inference.py para mandar el push de "riesgo de lluvia" solo
    cuando el riesgo pasa de False/None a True, no en cada lectura mientras se mantenga True
    (si no, un lote con riesgo sostenido por horas mandaría un push cada 30s con el poller)."""
    anterior = (
        db.query(Prediccion)
        .filter(Prediccion.id_lote == id_lote, Prediccion.riesgo_lluvia_proxima.isnot(None))
        .order_by(Prediccion.fecha_prediccion.desc())
        .first()
    )
    return anterior.riesgo_lluvia_proxima if anterior is not None else None


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


def _debe_notificar_push(severidad: str) -> bool:
    if not settings.fcm_enabled:
        return False
    minimo = SEVERITY_RANK.get(settings.fcm_min_severidad, 2)
    return SEVERITY_RANK.get(severidad, 0) >= minimo


def enviar_push_alerta(
    db: Session, id_lote: Optional[int], severidad: str, mensaje: str,
    titulo: Optional[str] = None, datos_extra: Optional[Dict[str, str]] = None,
) -> bool:
    """Envía notificación push (FCM) al dueño del lote, si la severidad lo amerita.

    Sigue la MISMA cadena de aislamiento por usuario que el resto del sistema:
    id_lote -> lotes_cafe.id_usuario -> dispositivos_usuario.id_usuario -- nunca manda la alerta
    a todos los dispositivos registrados, solo a los del dueño real de ESE lote (ver
    definicion_problema_kajve.md, Sección 5: RLS no protege este camino, hay que filtrar
    explícitamente en código, que es justo lo que hace este query).

    `titulo`: texto corto para el título de la notificación (ej. "Exceso de temperatura", ver
    rules.titulo_corto_para). Si no se manda, cae al formato genérico anterior.
    `datos_extra`: se mezcla en el payload `data` del push -- ahí va, entre otras cosas, el
    texto completo de la recomendación, para que la app móvil arme "alerta + recomendación" en
    una sola vista sin tener que llamar a otro endpoint.

    No lanza excepción si falla: la alerta ya quedó guardada en alertas/BD de todas formas."""
    if not _debe_notificar_push(severidad) or id_lote is None:
        return False

    lote = db.query(LoteCafe).filter(LoteCafe.id_lote == id_lote).first()
    if lote is None:
        return False

    dispositivos = (
        db.query(DispositivoUsuario)
        .filter(DispositivoUsuario.id_usuario == lote.id_usuario, DispositivoUsuario.activo.is_(True))
        .all()
    )
    tokens = [d.fcm_token for d in dispositivos]
    if not tokens:
        return False

    datos = {"id_lote": str(id_lote), "severidad": severidad}
    datos.update(datos_extra or {})

    resultado = fcm.enviar_push(
        tokens,
        titulo=titulo or f"Alerta {severidad.upper()} - {lote.nombre_lote}",
        cuerpo=mensaje,
        datos=datos,
    )

    # Limpieza automática: si FCM reporta un token como no-registrado/inválido (app
    # desinstalada, token rotado sin que el usuario haya vuelto a abrir la app), se desactiva
    # en vez de seguir intentando mandarle notificaciones que nunca van a llegar.
    if resultado["tokens_invalidos"]:
        db.query(DispositivoUsuario).filter(
            DispositivoUsuario.id_usuario == lote.id_usuario,
            DispositivoUsuario.fcm_token.in_(resultado["tokens_invalidos"]),
        ).update({"activo": False}, synchronize_session=False)
        db.commit()

    return resultado["enviados"] > 0


def debe_notificar_anomalia(db: Session, id_lote: int, tipo_anomalia: str) -> bool:
    """Cooldown por (id_lote, tipo_anomalia): evita ráfagas de push cuando la misma anomalía
    sigue presente en lecturas consecutivas (ej. el poller cada 30s durante horas). Mismo patrón
    que ultimo_riesgo_lluvia(): un SELECT fresco contra una tabla persistida real, ANTES de
    decidir notificar -- pero con un umbral de tiempo (FCM_COOLDOWN_MINUTOS) en vez de una
    transición booleana, porque una anomalía general no es un simple sí/no como el riesgo de
    lluvia."""
    anterior = (
        db.query(NotificacionPushAnomalia)
        .filter(
            NotificacionPushAnomalia.id_lote == id_lote,
            NotificacionPushAnomalia.tipo_anomalia == tipo_anomalia,
        )
        .first()
    )
    if anterior is None or anterior.fecha_ultimo_push is None:
        return True
    ahora_naive_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    minutos_transcurridos = (ahora_naive_utc - anterior.fecha_ultimo_push).total_seconds() / 60.0
    return minutos_transcurridos >= settings.fcm_cooldown_minutos


def registrar_push_anomalia(db: Session, id_lote: int, tipo_anomalia: str) -> None:
    """Marca 'se acaba de notificar esta anomalía para este lote' -- upsert por (id_lote,
    tipo_anomalia), mismo criterio que dispositivos.py::registrar_dispositivo: una sola fila por
    combinación, se actualiza en vez de acumular historial. Solo se llama cuando
    debe_notificar_anomalia() ya dio luz verde y se intentó el push -- si se llamara también
    cuando el cooldown bloquea el envío, cada ciclo del poller reiniciaría el cooldown sin haber
    notificado nada, y nunca se volvería a avisar."""
    existente = (
        db.query(NotificacionPushAnomalia)
        .filter(
            NotificacionPushAnomalia.id_lote == id_lote,
            NotificacionPushAnomalia.tipo_anomalia == tipo_anomalia,
        )
        .first()
    )
    ahora = datetime.now(timezone.utc).replace(tzinfo=None)
    if existente is not None:
        existente.fecha_ultimo_push = ahora
    else:
        db.add(NotificacionPushAnomalia(id_lote=id_lote, tipo_anomalia=tipo_anomalia, fecha_ultimo_push=ahora))
    db.commit()
