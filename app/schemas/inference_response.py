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
    # Algoritmo Genético (paso 7/11) -- riesgo de que llueva en las próximas
    # horas_anticipacion_lluvia horas, no si está lloviendo ahora (eso ya lo cubren las
    # reglas/alertas normales vía el sensor FC-37). None si no se pudo calcular (ver
    # app/services/rain_predictor.py).
    riesgo_lluvia_proxima: Optional[bool] = None
    horas_anticipacion_lluvia: Optional[int] = None


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
    notificacion_push_enviada: bool = False
    modelo_version: str
    fecha_inferencia: str
