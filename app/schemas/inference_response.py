#app/schemas/inference_response.py
from typing import Optional, List
from pydantic import BaseModel, Field


class InferenceResponse(BaseModel):
    id_inferencia: int
    id_lote: Optional[int]
    es_anomalia: bool
    score_anomalia: float
    nivel_severidad: str
    variables_contribuyentes: List[str]
    mensaje: str
    modelo_version: str = "0.1.0"
    fecha_inferencia: str
