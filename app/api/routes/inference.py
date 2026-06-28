import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.models.database import SessionLocal
from app.models.inference_record import InferenceRecord
from app.schemas.inference_request import InferenceRequest
from app.schemas.inference_response import InferenceResponse
from app.services.preprocessor import Preprocessor
from app.services.anomaly_detector import AnomalyDetector

router = APIRouter(tags=["inference"])
preprocessor = Preprocessor()
detector = AnomalyDetector()


@router.post("/anomalies/detect", response_model=InferenceResponse)
def detect_anomaly(request: InferenceRequest):
    try:
        features = preprocessor.transform(request.lecturas)
        is_anomaly, score, contrib = detector.predict(features)

        if score > 0:
            severity = "normal"
            message = "Patrón dentro de los rangos esperados"
        elif score > -0.1:
            severity = "advertencia"
            message = "Se observa una desviación leve"
        else:
            severity = "riesgo"
            message = "Patrón atípico de secado"

        if is_anomaly and (request.lecturas.get("lluvia", 0) >= 0.5 or request.lecturas.get("humedad_ambiental", 0) > 88):
            severity = "critico"
            message = "Riesgo crítico: humedad o lluvia inusuales"

        response = InferenceResponse(
            id_inferencia=0,
            id_lote=request.id_lote,
            es_anomalia=is_anomaly,
            score_anomalia=round(score, 3),
            nivel_severidad=severity,
            variables_contribuyentes=contrib or ["sin_datos"],
            mensaje=message,
            fecha_inferencia=datetime.utcnow().isoformat() + "Z",
        )

        db: Session = SessionLocal()
        try:
            record = InferenceRecord(
                id_lote=request.id_lote,
                tipo_proceso=request.tipo_proceso,
                payload_entrada=json.dumps(request.lecturas),
                es_anomalia=is_anomaly,
                score_anomalia=response.score_anomalia,
                nivel_severidad=severity,
                mensaje=message,
                modelo_version="0.1.0",
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            response.id_inferencia = record.id
        finally:
            db.close()

        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
