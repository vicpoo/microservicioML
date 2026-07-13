# Archivo: app/api/routes/history.py
# Carpeta: microservicioMLL/app/api/routes/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.security import verificar_api_key
from app.models.alertas import Alerta
from app.models.database import SessionLocal
from app.models.lotes_cafe import LoteCafe
from app.models.predicciones import Prediccion
from app.models.recomendaciones import Recomendacion

# OPCIONAL: tu API móvil puede leer alertas/predicciones/recomendaciones directo de Neon
# (son tablas normales, el MLL ya escribió ahí) sin pasar por aquí. Estos endpoints quedan
# disponibles por si prefieres que tu API móvil se los pida al MLL en vez de tener su propia
# consulta a esas 3 tablas. Protegidos con la misma X-Internal-Api-Key que usa el Gestor.
router = APIRouter(tags=["history"], dependencies=[Depends(verificar_api_key)])


def _verificar_dueno(db: Session, id_lote: int, id_usuario: int) -> None:
    lote = db.query(LoteCafe).filter(LoteCafe.id_lote == id_lote).first()
    if lote is None:
        raise HTTPException(status_code=404, detail="Lote no encontrado")
    if lote.id_usuario != id_usuario:
        raise HTTPException(status_code=403, detail="El lote no pertenece a este usuario")


@router.get("/anomalies")
def listar_alertas(
    id_usuario: int = Query(description="Usuario dueño de la sesión (lo resuelve quien llama, ej. tu API móvil)"),
    id_lote: Optional[int] = Query(default=None),
    solo_no_atendidas: bool = Query(default=False),
    limit: int = Query(default=20, le=200),
    offset: int = Query(default=0, ge=0),
):
    db: Session = SessionLocal()
    try:
        if id_lote is not None:
            _verificar_dueno(db, id_lote, id_usuario)
            query = db.query(Alerta).filter(Alerta.id_lote == id_lote)
        else:
            query = db.query(Alerta).join(LoteCafe, Alerta.id_lote == LoteCafe.id_lote).filter(
                LoteCafe.id_usuario == id_usuario
            )
        if solo_no_atendidas:
            query = query.filter(Alerta.atendida.is_(False))
        registros = query.order_by(Alerta.fecha_generada.desc()).offset(offset).limit(limit).all()
        return [
            {
                "id_alerta": r.id_alerta,
                "id_lote": r.id_lote,
                "id_sensor": r.id_sensor,
                "tipo_alerta": r.tipo_alerta,
                "mensaje": r.mensaje,
                "nivel_severidad": r.nivel_severidad,
                "atendida": r.atendida,
                "fecha_generada": r.fecha_generada.isoformat() if r.fecha_generada else None,
            }
            for r in registros
        ]
    finally:
        db.close()


@router.get("/anomalies/{id_lote}/predicciones")
def listar_predicciones(
    id_lote: int,
    id_usuario: int = Query(description="Usuario dueño de la sesión (lo resuelve quien llama, ej. tu API móvil)"),
    limit: int = Query(default=20, le=200),
):
    db: Session = SessionLocal()
    try:
        _verificar_dueno(db, id_lote, id_usuario)
        registros = (
            db.query(Prediccion)
            .filter(Prediccion.id_lote == id_lote)
            .order_by(Prediccion.fecha_prediccion.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id_prediccion": r.id_prediccion,
                "id_lote": r.id_lote,
                "tiempo_estimado_horas": float(r.tiempo_estimado_horas) if r.tiempo_estimado_horas is not None else None,
                "calidad_estimada": r.calidad_estimada,
                "confianza": float(r.confianza) if r.confianza is not None else None,
                "fecha_prediccion": r.fecha_prediccion.isoformat() if r.fecha_prediccion else None,
            }
            for r in registros
        ]
    finally:
        db.close()


@router.get("/anomalies/{id_lote}/recomendaciones")
def listar_recomendaciones(
    id_lote: int,
    id_usuario: int = Query(description="Usuario dueño de la sesión (lo resuelve quien llama, ej. tu API móvil)"),
    limit: int = Query(default=20, le=200),
):
    db: Session = SessionLocal()
    try:
        _verificar_dueno(db, id_lote, id_usuario)
        registros = (
            db.query(Recomendacion)
            .filter(Recomendacion.id_lote == id_lote)
            .order_by(Recomendacion.fecha_generada.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id_recomendacion": r.id_recomendacion,
                "id_lote": r.id_lote,
                "texto": r.texto,
                "origen": r.origen,
                "fecha_generada": r.fecha_generada.isoformat() if r.fecha_generada else None,
            }
            for r in registros
        ]
    finally:
        db.close()