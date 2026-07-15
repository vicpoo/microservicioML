# Archivo: app/schemas/internal_events.py
# Carpeta: microservicioMLL/app/schemas/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LecturaNuevaEvent(BaseModel):
    """Lo que manda el Gestor cuando ya guardó una lectura nueva en lecturas_ambientales."""
    id_lote: int
    id_lectura: Optional[int] = Field(default=None, description="Si no se manda, se toma la lectura más reciente del lote")


class ResultadoRealEvent(BaseModel):
    """Lo que reporta el productor (vía Gestor) al finalizar el secado de un lote (RNF-19):
    la etiqueta real que retroalimenta el reentrenamiento, en vez de solo datos sintéticos."""
    calidad_real: Literal["excelente", "buena", "regular", "baja"] = Field(
        description="Calidad final del lote, evaluada por el productor (Cuadro 8/11 del Documento de Calidad del Café)"
    )
    tiempo_real_horas: Optional[float] = Field(
        default=None,
        description="Horas reales totales de secado. Si no se manda, se calcula desde fecha_inicio_secado del lote hasta ahora.",
    )


class ResultadoRealResponse(BaseModel):
    id_retroalimentacion: int
    mensaje: str