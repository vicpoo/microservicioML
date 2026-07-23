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
from app.models.reportes_lote import ReporteLote
from NLP.buscar_reportes import TOP_N_RESULTADOS_DEFAULT, buscar_reportes
from NLP.generar_reporte import generar_reporte_lote
from NLP.recopilar_datos_reporte import recopilar_datos_lote
from NLP.registrar_reporte import guardar_reporte, historial_reportes

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


@router.get("/anomalies/reportes/buscar")
def buscar_reportes_endpoint(
    id_usuario: int = Query(description="Usuario dueño de la sesión (lo resuelve quien llama, ej. tu API móvil)"),
    query: str = Query(min_length=1, description="Texto libre a buscar, ej. 'lluvia crítica' o 'secado estancado'"),
    top_n: int = Query(default=TOP_N_RESULTADOS_DEFAULT, le=50),
):
    """Paso 3 de la opción A (buscador de historial, ver NLP/README.md): busca con BM25 sobre
    TODOS los reportes ya generados (GET .../reporte) de TODOS los lotes de `id_usuario` -- no
    de un solo lote, a diferencia del resto de endpoints de este archivo. El aislamiento por
    usuario aquí es más simple que `_verificar_dueno` (que valida un id_lote puntual): el JOIN
    contra `lotes_cafe.id_usuario` ya arma el corpus SOLO con reportes de lotes de ese usuario,
    así que no hay manera de que aparezca un reporte ajeno en los resultados.

    Ruta estática (no `/anomalies/{id_lote}/...`) a propósito: esta búsqueda es sobre el
    historial completo del usuario, no sobre un lote en particular."""
    db: Session = SessionLocal()
    try:
        filas = (
            db.query(ReporteLote)
            .join(LoteCafe, ReporteLote.id_lote == LoteCafe.id_lote)
            .filter(LoteCafe.id_usuario == id_usuario)
            .all()
        )
        if not filas:
            return []

        corpus = [(f.id_reporte, f.reporte_texto) for f in filas]
        por_id = {f.id_reporte: f for f in filas}
        resultados = buscar_reportes(corpus, query, top_n=top_n)
        return [
            {
                "id_reporte": r.id_reporte,
                "id_lote": por_id[r.id_reporte].id_lote,
                "score": round(r.score, 4),
                "reporte_texto": r.texto,
                "fecha_generado": (
                    por_id[r.id_reporte].fecha_generado.isoformat()
                    if por_id[r.id_reporte].fecha_generado else None
                ),
            }
            for r in resultados
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
                "calidad_estimada": float(r.calidad_estimada) if r.calidad_estimada is not None else None,
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


@router.get("/anomalies/{id_lote}/reporte")
def obtener_reporte(
    id_lote: int,
    id_usuario: int = Query(description="Usuario dueño de la sesión (lo resuelve quien llama, ej. tu API móvil)"),
):
    """PLN (NLP/, pasos 1-4): reporte en lenguaje natural del lote -- combina lo que ya sabe el
    resto del sistema (alertas, predicciones, recomendaciones) en un solo texto legible, en vez
    de que la app móvil tenga que armarlo ella misma a partir de 3 endpoints distintos. Cada
    llamada genera el texto AL MOMENTO (siempre refleja el estado actual) y además queda
    guardada en `reportes_lote` -- mismo criterio que predicciones/alertas/recomendaciones: se
    acumula historial, no se sobrescribe (ver GET .../reportes, plural, para verlo)."""
    db: Session = SessionLocal()
    try:
        _verificar_dueno(db, id_lote, id_usuario)
        datos = recopilar_datos_lote(db, id_lote)
        # _verificar_dueno ya garantiza que el lote existe -- si datos viniera None aquí sería
        # un bug interno, no un 404 de "lote no encontrado" (ese caso ya se descartó arriba).
        assert datos is not None, "lote confirmado por _verificar_dueno pero recopilar_datos_lote devolvió None"
        texto = generar_reporte_lote(datos)
        id_reporte = guardar_reporte(db, id_lote, texto)
        return {
            "id_reporte": id_reporte,
            "id_lote": id_lote,
            "reporte_texto": texto,
            "fecha_generado": datos.fecha_generado.isoformat(),
        }
    finally:
        db.close()


@router.get("/anomalies/{id_lote}/reportes")
def listar_reportes(
    id_lote: int,
    id_usuario: int = Query(description="Usuario dueño de la sesión (lo resuelve quien llama, ej. tu API móvil)"),
    limit: int = Query(default=10, le=200),
):
    """Historial de reportes ya generados para este lote (los que quedaron guardados por
    GET .../reporte, singular), más reciente primero -- para ver cómo cambió el reporte de un
    lote con el tiempo sin tener que regenerarlo."""
    db: Session = SessionLocal()
    try:
        _verificar_dueno(db, id_lote, id_usuario)
        return historial_reportes(db, id_lote, limit=limit)
    finally:
        db.close()