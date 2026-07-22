# Archivo: app/api/routes/internal.py
# Carpeta: microservicioMLL/app/api/routes/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.routes.inference import ejecutar_pipeline
from app.core.security import verificar_api_key
from app.models.database import SessionLocal
from app.models.lecturas_ambientales import LecturaAmbiental
from app.models.lotes_cafe import LoteCafe
from app.models.retroalimentacion_ml import RetroalimentacionML
from app.schemas.inference_response import InferenceResponse
from app.schemas.internal_events import LecturaNuevaEvent, ResultadoRealEvent, ResultadoRealResponse
from app.services import poller
from app.services.lectura_utils import calcular_horas_transcurridas, construir_features
from ML import monitoreo

# Lo llama SOLO el Servicio Gestor, justo después de escribir una lectura en Neon.
# No hay concepto de "dueño" que validar aquí: el Gestor es un servicio de confianza,
# no un usuario final. El id_usuario para las validaciones de dueño en history.py lo manda
# quien llama al MLL (Gestor o API móvil), como servicio interno de confianza.
router = APIRouter(prefix="/internal", tags=["internal"], dependencies=[Depends(verificar_api_key)])


@router.post("/lecturas/nuevas", response_model=InferenceResponse)
def procesar_lectura_nueva(evento: LecturaNuevaEvent):
    """El Gestor solo avisa 'hay algo nuevo en el lote X'; el MLL va y lee el dato real de
    Neon (así el Gestor no necesita saber nada de las 6 variables ni del formato del modelo)."""
    db: Session = SessionLocal()
    try:
        lote = db.query(LoteCafe).filter(LoteCafe.id_lote == evento.id_lote).first()
        if lote is None:
            raise HTTPException(status_code=404, detail="Lote no encontrado")

        query = db.query(LecturaAmbiental).filter(LecturaAmbiental.id_lote == evento.id_lote)
        if evento.id_lectura is not None:
            lectura = query.filter(LecturaAmbiental.id_lectura == evento.id_lectura).first()
        else:
            lectura = query.order_by(LecturaAmbiental.timestamp.desc()).first()
        if lectura is None:
            raise HTTPException(status_code=404, detail="No hay lecturas para ese lote en lecturas_ambientales")

        tipo_proceso = (lote.tipo_proceso or "lavado").lower()
        horas_transcurridas = calcular_horas_transcurridas(lote)
        features = construir_features(lectura)

        presion_hpa = float(lectura.presion_hpa) if lectura.presion_hpa is not None else None
        respuesta = ejecutar_pipeline(
            db, lote, evento.id_lote, tipo_proceso, lectura.id_sensor, features, horas_transcurridas,
            guardar_lectura=False,  # el Gestor ya la guardó, no la duplicamos
            presion_hpa=presion_hpa,
        )
        # Avanza el cursor compartido con app/services/poller.py: esta lectura ya se procesó
        # por el webhook, así que cuando el poller le toque revisar este rango la salta (si
        # no, la volvería a procesar y duplicaría predicción/alerta/push para el mismo dato).
        poller.marcar_procesada(db, lectura.id_lectura)
        return respuesta
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


@router.post("/lotes/{id_lote}/resultado-real", response_model=ResultadoRealResponse, status_code=201)
def registrar_resultado_real(id_lote: int, evento: ResultadoRealEvent):
    """RNF-19: captura la etiqueta real que reporta el productor (vía Gestor) al finalizar el
    secado de un lote — calidad final y tiempo real. Se guarda en retroalimentacion_ml, separada
    del dataset sintético; scripts/train_models.py la combina al reentrenar cuando hay datos."""
    db: Session = SessionLocal()
    try:
        lote = db.query(LoteCafe).filter(LoteCafe.id_lote == id_lote).first()
        if lote is None:
            raise HTTPException(status_code=404, detail="Lote no encontrado")

        ultima = (
            db.query(LecturaAmbiental)
            .filter(LecturaAmbiental.id_lote == id_lote)
            .order_by(LecturaAmbiental.timestamp.desc())
            .first()
        )
        if ultima is None:
            raise HTTPException(status_code=404, detail="No hay lecturas_ambientales para ese lote; no se puede construir el ejemplo etiquetado")

        if evento.tiempo_real_horas is not None:
            tiempo_real_horas = evento.tiempo_real_horas
        elif lote.fecha_inicio_secado:
            inicio = lote.fecha_inicio_secado
            if inicio.tzinfo is None:
                inicio = inicio.replace(tzinfo=timezone.utc)
            tiempo_real_horas = max((datetime.now(timezone.utc) - inicio).total_seconds() / 3600.0, 0.0)
        else:
            raise HTTPException(status_code=422, detail="tiempo_real_horas es obligatorio: el lote no tiene fecha_inicio_secado")

        registro = RetroalimentacionML(
            id_lote=id_lote,
            tipo_proceso=(lote.tipo_proceso or "lavado").lower(),
            temperatura_grano=ultima.temperatura_grano,
            temperatura_ambiental=ultima.temperatura,
            humedad_grano=ultima.humedad_grano,
            lluvia_detectada=ultima.lluvia_detectada,
            luz=ultima.luz,
            tiempo_real_horas=round(tiempo_real_horas, 2),
            calidad_real=evento.calidad_real,
        )
        db.add(registro)
        db.commit()
        db.refresh(registro)
        return ResultadoRealResponse(id_retroalimentacion=registro.id_retroalimentacion, mensaje="Resultado real registrado")
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


@router.get("/monitoreo/salud")
def salud_modelos(dias_alertas: int = 7) -> Dict[str, Any]:
    """Paso 12 (monitoreo y reentrenamiento): compara predicciones ya hechas contra la
    retroalimentación real reportada por productores, vigila la tasa de alertas reciente, y
    dice si ya hay datos suficientes para reentrenar -- ver ML/monitoreo.py para el detalle de
    cada métrica. Pensado para un cron/dashboard del Gestor, no para la app móvil."""
    db: Session = SessionLocal()
    try:
        return monitoreo.resumen_salud(db, dias_alertas=dias_alertas)
    finally:
        db.close()