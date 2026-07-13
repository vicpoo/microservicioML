# Archivo: app/api/routes/inference.py
# Carpeta: microservicioMLL/app/api/routes/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import verificar_api_key
from app.models.database import SessionLocal
from app.models.lecturas_ambientales import LecturaAmbiental
from app.models.lotes_cafe import LoteCafe
from app.schemas.inference_request import InferenceRequest
from app.schemas.inference_response import InferenceResponse, PrediccionOut, RecomendacionOut
from app.services import notifier
from app.services.anomaly_detector import AnomalyDetector
from app.services.predictor import Predictor
from app.services.preprocessor import Preprocessor
from app.services.recommender import Recommender

# Endpoints en este archivo: llamados por servicios internos (Gestor, o tú mismo probando
# con curl), nunca por la app móvil directo. Se protegen con X-Internal-Api-Key.
router = APIRouter(tags=["inference"], dependencies=[Depends(verificar_api_key)])
settings = get_settings()

preprocessor = Preprocessor()
detector = AnomalyDetector()
predictor = Predictor()
recommender = Recommender()

MENSAJES_SEVERIDAD = {
    "normal": "Patrón dentro de los rangos esperados para este proceso.",
    "advertencia": "Se observa una desviación leve respecto al rango ideal.",
    "riesgo": "Patrón de riesgo: se recomienda atender el lote pronto.",
    "critico": "Riesgo crítico: se requiere atención inmediata.",
}

# Solo estas severidades generan una alerta real (tabla alertas + correo). advertencia y
# normal solo alimentan predicciones/recomendaciones, no "molestan" al usuario.
SEVERIDADES_QUE_ALERTAN = {"riesgo", "critico"}


def _contexto_lote(db: Session, id_lote: Optional[int], id_usuario: int, tipo_proceso_fallback: Optional[str], id_sensor_fallback: Optional[int]):
    lote = None
    if id_lote is not None:
        lote = db.query(LoteCafe).filter(LoteCafe.id_lote == id_lote).first()

    if lote is not None:
        if lote.id_usuario != id_usuario:
            raise HTTPException(status_code=403, detail="El lote no pertenece a este usuario")
        tipo_proceso = (lote.tipo_proceso or tipo_proceso_fallback or "lavado").lower()
        id_sensor = id_sensor_fallback or lote.id_sensor
        if lote.fecha_inicio_secado:
            inicio = lote.fecha_inicio_secado
            if inicio.tzinfo is None:
                inicio = inicio.replace(tzinfo=timezone.utc)
            horas_transcurridas = max((datetime.now(timezone.utc) - inicio).total_seconds() / 3600.0, 0.0)
        else:
            horas_transcurridas = 0.0
    else:
        tipo_proceso = (tipo_proceso_fallback or "lavado").lower()
        id_sensor = id_sensor_fallback
        horas_transcurridas = 0.0

    return lote, tipo_proceso, id_sensor, horas_transcurridas


def _contexto_historico(db: Session, id_lote: Optional[int], features: Dict[str, float]):
    if id_lote is None:
        return None, None
    ultima = (
        db.query(LecturaAmbiental)
        .filter(LecturaAmbiental.id_lote == id_lote)
        .order_by(LecturaAmbiental.timestamp.desc())
        .first()
    )
    delta_temp_reciente = None
    if ultima is not None and ultima.temperatura_grano is not None:
        delta_temp_reciente = abs(features["temperatura_grano"] - float(ultima.temperatura_grano))

    hace_24h = datetime.utcnow() - timedelta(hours=24)
    lectura_24h = (
        db.query(LecturaAmbiental)
        .filter(LecturaAmbiental.id_lote == id_lote, LecturaAmbiental.timestamp <= hace_24h)
        .order_by(LecturaAmbiental.timestamp.desc())
        .first()
    )
    delta_humedad_grano_24h = None
    if lectura_24h is not None and lectura_24h.humedad_grano is not None:
        delta_humedad_grano_24h = float(lectura_24h.humedad_grano) - features["humedad_grano"]

    return delta_temp_reciente, delta_humedad_grano_24h


def ejecutar_pipeline(
    db: Session,
    lote: Optional[LoteCafe],
    id_lote_solicitado: Optional[int],
    tipo_proceso: str,
    id_sensor: Optional[int],
    features: Dict[str, float],
    horas_transcurridas: float,
    guardar_lectura: bool,
) -> InferenceResponse:
    """Corazón del microservicio: dado un vector de 6 lecturas ya resuelto (venga del body
    de /detect o de una fila real de lecturas_ambientales), corre detección + predicción +
    recomendación y persiste todo. Lo usan tanto el endpoint manual/testing (POST /detect)
    como el endpoint que dispara el Gestor (POST /internal/lecturas/nuevas)."""
    delta_temp_reciente, delta_humedad_grano_24h = _contexto_historico(db, id_lote_solicitado, features)

    resultado = detector.predict(
        tipo_proceso, features,
        delta_temp_reciente=delta_temp_reciente,
        delta_humedad_grano_24h=delta_humedad_grano_24h,
    )
    prediccion = predictor.predecir(tipo_proceso, features, horas_transcurridas)
    recomendaciones = recommender.generar(resultado["alertas"])
    mensaje = MENSAJES_SEVERIDAD.get(resultado["severidad"], MENSAJES_SEVERIDAD["normal"])
    if resultado["alertas"]:
        mensaje = f"{mensaje} {resultado['alertas'][0]['mensaje']}"

    id_inferencia = notifier.registrar_inferencia(db, features, resultado["severidad"], mensaje)

    if guardar_lectura and lote is not None and id_sensor is not None:
        db.add(LecturaAmbiental(
            id_sensor=id_sensor,
            id_lote=lote.id_lote,
            temperatura=features["temperatura_ambiental"],
            humedad=features["humedad_ambiental"],
            humedad_grano=features["humedad_grano"],
            temperatura_grano=features["temperatura_grano"],
            luz=features["luz"],
            lluvia=features["lluvia"],
        ))
        db.commit()

    alerta_generada = False
    id_alerta = None
    email_enviado = False

    if lote is not None:
        # Predicciones y recomendaciones se generan SIEMPRE que hay lectura (usan todos los
        # sensores, no solo cuando hay anomalía). La alerta (tabla alertas + correo) solo
        # dispara para severidad riesgo/critico: no queremos avisar por cada advertencia leve.
        if resultado["severidad"] in SEVERIDADES_QUE_ALERTAN:
            id_alerta = notifier.registrar_alerta(
                db, lote.id_lote, id_sensor, resultado["tipo_principal"], mensaje, resultado["severidad"]
            )
            alerta_generada = True
            email_enviado = notifier.enviar_email_alerta(
                lote.id_lote, resultado["severidad"], mensaje, recomendaciones
            )
        notifier.registrar_recomendaciones(db, lote.id_lote, recomendaciones)
        id_modelo = notifier.get_or_create_modelo(db)
        notifier.registrar_prediccion(
            db, lote.id_lote, id_modelo,
            prediccion["tiempo_estimado_horas"], prediccion["calidad_estimada"], prediccion["confianza"],
        )
    elif id_lote_solicitado is not None:
        mensaje += " (Aviso: id_lote no existe en lotes_cafe; no se generó alerta/recomendación/predicción persistida, solo la bitácora.)"

    return InferenceResponse(
        id_inferencia=id_inferencia,
        id_lote=lote.id_lote if lote is not None else id_lote_solicitado,
        es_anomalia=resultado["es_anomalia"],
        nivel_severidad=resultado["severidad"],
        score_isolation_forest=resultado["score_isolation_forest"],
        confianza_ml=resultado["confianza_ml"],
        variables_contribuyentes=resultado["variables_contribuyentes"],
        mensaje=mensaje,
        recomendaciones=[RecomendacionOut(**r) for r in recomendaciones],
        prediccion=PrediccionOut(**prediccion),
        alerta_generada=alerta_generada,
        id_alerta=id_alerta,
        notificacion_email_enviada=email_enviado,
        modelo_version=settings.modelo_version,
        fecha_inferencia=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/anomalies/detect", response_model=InferenceResponse)
def detect_anomaly(request: InferenceRequest):
    """Endpoint MANUAL / de pruebas: le mandas las 6 lecturas directo en el body (curl,
    Postman, tests). En producción, el disparador real es /internal/lecturas/nuevas."""
    db: Session = SessionLocal()
    try:
        lote, tipo_proceso, id_sensor, horas_transcurridas = _contexto_lote(
            db, request.id_lote, request.id_usuario, request.tipo_proceso, request.id_sensor
        )
        features = preprocessor.transform(request.lecturas)
        return ejecutar_pipeline(
            db, lote, request.id_lote, tipo_proceso, id_sensor, features, horas_transcurridas,
            guardar_lectura=request.guardar_lectura,
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()