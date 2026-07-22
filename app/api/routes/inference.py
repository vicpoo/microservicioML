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
from app.services import notifier, rules
from app.services.anomaly_detector import AnomalyDetector
from app.services.predictor import Predictor
from app.services.preprocessor import Preprocessor
from app.services.rain_predictor import PredictorLluvia
from app.services.recommender import Recommender

# Endpoints en este archivo: llamados por servicios internos (Gestor, o tú mismo probando
# con curl), nunca por la app móvil directo. Se protegen con X-Internal-Api-Key.
router = APIRouter(tags=["inference"], dependencies=[Depends(verificar_api_key)])
settings = get_settings()

preprocessor = Preprocessor()
detector = AnomalyDetector()
predictor = Predictor()
predictor_lluvia = PredictorLluvia()
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
    # humedad_grano en la tabla real es el CRUDO del ADC, no un porcentaje: hay que
    # convertir ambos extremos con rules.humedad_grano_raw_a_porcentaje antes de
    # restar. Si el sensor no está calibrado (rules.RAW_GRANO_HUMEDO/SECO en None),
    # la conversión regresa None y aquí se propaga None (la regla se omite, no se
    # inventa un delta con unidades incorrectas).
    delta_humedad_grano_24h_pct = None
    if lectura_24h is not None and lectura_24h.humedad_grano is not None:
        pct_antiguo = rules.humedad_grano_raw_a_porcentaje(float(lectura_24h.humedad_grano))
        pct_actual = rules.humedad_grano_raw_a_porcentaje(features.get("humedad_grano"))
        if pct_antiguo is not None and pct_actual is not None:
            delta_humedad_grano_24h_pct = pct_antiguo - pct_actual

    return delta_temp_reciente, delta_humedad_grano_24h_pct


def ejecutar_pipeline(
    db: Session,
    lote: Optional[LoteCafe],
    id_lote_solicitado: Optional[int],
    tipo_proceso: str,
    id_sensor: Optional[int],
    features: Dict[str, float],
    horas_transcurridas: float,
    guardar_lectura: bool,
    presion_hpa: Optional[float] = None,
) -> InferenceResponse:
    """Corazón del microservicio: dado un vector de 6 lecturas ya resuelto (venga del body
    de /detect o de una fila real de lecturas_ambientales), corre detección + predicción +
    recomendación y persiste todo. Lo usan tanto el endpoint manual/testing (POST /detect)
    como el endpoint que dispara el Gestor (POST /internal/lecturas/nuevas) y el poller.

    presion_hpa es aparte de `features` (no es una de las 6 variables estándar del resto del
    pipeline) porque solo la usa el Algoritmo Genético de lluvia -- ver
    app/services/rain_predictor.py."""
    delta_temp_reciente, delta_humedad_grano_24h_pct = _contexto_historico(db, id_lote_solicitado, features)

    resultado = detector.predict(
        tipo_proceso, features,
        delta_temp_reciente=delta_temp_reciente,
        delta_humedad_grano_24h_pct=delta_humedad_grano_24h_pct,
    )
    prediccion = predictor.predecir(tipo_proceso, features, horas_transcurridas)

    riesgo_lluvia = None
    if lote is not None:
        riesgo_lluvia = predictor_lluvia.predecir(
            db, lote.id_lote, presion_hpa, features.get("luz", 0.0),
            ahora=datetime.now(timezone.utc),
        )
    prediccion["riesgo_lluvia_proxima"] = riesgo_lluvia["riesgo_lluvia_proxima"] if riesgo_lluvia else None
    prediccion["horas_anticipacion_lluvia"] = riesgo_lluvia["horas_anticipacion"] if riesgo_lluvia else None

    recomendaciones = recommender.generar(resultado["alertas"])
    mensaje = MENSAJES_SEVERIDAD.get(resultado["severidad"], MENSAJES_SEVERIDAD["normal"])
    if resultado["alertas"]:
        mensaje = f"{mensaje} {resultado['alertas'][0]['mensaje']}"

    id_inferencia = notifier.registrar_inferencia(db, features, resultado["severidad"], mensaje)

    if guardar_lectura and lote is not None and id_sensor is not None:
        # Este endpoint es manual/de pruebas (curl, Postman): no conoce presion_hpa,
        # altitud_m ni el valor crudo de lluvia_analog (eso lo manda el firmware real
        # vía el Gestor). Se guarda lo que sí se tiene; lluvia_detectada se resuelve
        # del mismo umbral que ya usa el motor de reglas (rules.LLUVIA_UMBRAL).
        db.add(LecturaAmbiental(
            id_sensor=id_sensor,
            id_lote=lote.id_lote,
            temperatura=features["temperatura_ambiental"],
            humedad_grano=features["humedad_grano"],
            temperatura_grano=features["temperatura_grano"],
            luz=features["luz"],
            lluvia_detectada=features["lluvia"] >= rules.LLUVIA_UMBRAL,
        ))
        db.commit()

    alerta_generada = False
    id_alerta = None
    email_enviado = False
    push_enviado = False

    if lote is not None:
        # Predicciones y recomendaciones se generan SIEMPRE que hay lectura (usan todos los
        # sensores, no solo cuando hay anomalía). La alerta FORMAL (tabla alertas + correo)
        # solo dispara para severidad riesgo/critico: no queremos saturar de correos por cada
        # advertencia leve.
        if resultado["severidad"] in SEVERIDADES_QUE_ALERTAN:
            id_alerta = notifier.registrar_alerta(
                db, lote.id_lote, id_sensor, resultado["tipo_principal"], mensaje, resultado["severidad"]
            )
            alerta_generada = True
            email_enviado = notifier.enviar_email_alerta(
                lote.id_lote, resultado["severidad"], mensaje, recomendaciones
            )

        # El push SÍ es independiente de esa alerta formal: dispara en CUALQUIER anomalía que
        # el ML detecte (es_anomalia=True implica severidad >= "advertencia", ver
        # AnomalyDetector.predict), no solo riesgo/critico -- es el canal pensado justo para
        # avisar al instante apenas se detecta algo, y FCM_MIN_SEVERIDAD (default:
        # "advertencia") controla desde qué nivel. No lanza excepción si falla, ni depende de
        # que se haya creado una fila en `alertas`.
        #
        # Título corto (rules.titulo_corto_para, ej. "Exceso de temperatura") + el texto de
        # recomendación correspondiente van en el push -- no solo la severidad genérica -- para
        # que la app móvil pueda mostrar "alerta + recomendación" de una sola notificación, sin
        # tener que llamar a otro endpoint aparte.
        if resultado["es_anomalia"]:
            tipo_principal = resultado["tipo_principal"]
            recomendacion_texto = next(
                (r["texto"] for r in recomendaciones if r["tipo"] == tipo_principal),
                recomendaciones[0]["texto"] if recomendaciones else mensaje,
            )
            push_enviado = notifier.enviar_push_alerta(
                db, lote.id_lote, resultado["severidad"], recomendacion_texto,
                titulo=rules.titulo_corto_para(tipo_principal),
                datos_extra={"tipo": tipo_principal, "recomendacion": recomendacion_texto},
            )

        # Push de riesgo de lluvia: SOLO en la transición False/None -> True, no en cada
        # lectura mientras el riesgo se mantenga True (el poller revisa cada 30s -- sin este
        # debounce, un riesgo sostenido por horas mandaría decenas de push idénticos). Se
        # compara contra la última predicción guardada de este lote, ANTES de registrar la
        # nueva (por eso se llama antes de registrar_prediccion).
        if prediccion["riesgo_lluvia_proxima"]:
            riesgo_anterior = notifier.ultimo_riesgo_lluvia(db, lote.id_lote)
            if not riesgo_anterior:
                texto_lluvia = (
                    f"Riesgo de lluvia en las próximas {prediccion['horas_anticipacion_lluvia']} horas "
                    "según el modelo. Considera cubrir el lote preventivamente."
                )
                notifier.enviar_push_alerta(
                    db, lote.id_lote, "advertencia", texto_lluvia,
                    titulo=rules.titulo_corto_para("riesgo_lluvia_proxima"),
                    datos_extra={"tipo": "riesgo_lluvia_proxima", "recomendacion": texto_lluvia},
                )

        notifier.registrar_recomendaciones(db, lote.id_lote, recomendaciones)
        id_modelo = notifier.get_or_create_modelo(db)
        notifier.registrar_prediccion(
            db, lote.id_lote, id_modelo,
            prediccion["tiempo_estimado_horas"], prediccion["calidad_estimada"], prediccion["confianza"],
            prediccion["riesgo_lluvia_proxima"], prediccion["horas_anticipacion_lluvia"],
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
        notificacion_push_enviada=push_enviado,
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
        # presion_hpa no es una de las 5 claves estándar de Preprocessor (solo la usa el AG de
        # lluvia) -- si el llamador la mandó en el body, se toma de ahí.
        presion_hpa = request.lecturas.get("presion_hpa")
        return ejecutar_pipeline(
            db, lote, request.id_lote, tipo_proceso, id_sensor, features, horas_transcurridas,
            guardar_lectura=request.guardar_lectura, presion_hpa=presion_hpa,
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()