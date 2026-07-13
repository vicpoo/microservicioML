#app/schemas/inference_response.py
from typing import List, Optional

from pydantic import BaseModel


class RecomendacionOut(BaseModel):
    tipo: str
    texto: str


class PrediccionOut(BaseModel):
    tiempo_estimado_horas: Optional[float] = None
    calidad_estimada: Optional[str] = None
    confianza: Optional[float] = None


class InferenceResponse(BaseModel):
    id_inferencia: int
    id_lote: Optional[int]
    es_anomalia: bool
    nivel_severidad: str  # normal | advertencia | riesgo | critico
    score_isolation_forest: float
    confianza_ml: float
    variables_contribuyentes: List[str]
    mensaje: str
    recomendaciones: List[RecomendacionOut]
    prediccion: PrediccionOut
    alerta_generada: bool
    id_alerta: Optional[int] = None
    notificacion_email_enviada: bool = False
    modelo_version: str
    fecha_inferencia: str
