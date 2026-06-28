from pydantic import BaseModel, Field
from typing import Dict, Optional


class InferenceRequest(BaseModel):
    id_lote: Optional[int] = None
    tipo_proceso: Optional[str] = Field(default=None, description="lavado, honey o natural")
    timestamp: Optional[str] = None
    lecturas: Dict[str, float]
