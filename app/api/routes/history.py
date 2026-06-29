#app/api/routes/history.py
from typing import List, Optional

from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from app.models.database import SessionLocal
from app.models.inference_record import InferenceRecord

router = APIRouter(tags=["history"])


@router.get("/anomalies")
def get_history(
    id_lote: Optional[int] = Query(default=None),
    limit: int = Query(default=10, le=100),
    offset: int = Query(default=0, ge=0),
):
    db: Session = SessionLocal()
    try:
        query = db.query(InferenceRecord)
        if id_lote is not None:
            query = query.filter(InferenceRecord.id_lote == id_lote)
        records = query.order_by(InferenceRecord.fecha_inferencia.desc()).offset(offset).limit(limit).all()
        return [
            {
                "id_inferencia": record.id,
                "id_lote": record.id_lote,
                "tipo_proceso": record.tipo_proceso,
                "es_anomalia": record.es_anomalia,
                "score_anomalia": record.score_anomalia,
                "nivel_severidad": record.nivel_severidad,
                "mensaje": record.mensaje,
                "modelo_version": record.modelo_version,
                "fecha_inferencia": record.fecha_inferencia.isoformat() if record.fecha_inferencia else None,
            }
            for record in records
        ]
    finally:
        db.close()
