# Archivo: app/schemas/internal_events.py
# Carpeta: microservicioMLL/app/schemas/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from typing import Optional

from pydantic import BaseModel, Field


class LecturaNuevaEvent(BaseModel):
    """Lo que manda el Gestor cuando ya guardó una lectura nueva en lecturas_ambientales."""
    id_lote: int
    id_lectura: Optional[int] = Field(default=None, description="Si no se manda, se toma la lectura más reciente del lote")