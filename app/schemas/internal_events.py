# Archivo: app/schemas/internal_events.py
# Carpeta: microservicioMLL/app/schemas/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from typing import Optional

from pydantic import BaseModel, Field


class LecturaNuevaEvent(BaseModel):
    """Lo que manda el Gestor cuando ya guardó una lectura nueva en lecturas_ambientales."""
    id_lote: int
    id_lectura: Optional[int] = Field(default=None, description="Si no se manda, se toma la lectura más reciente del lote")


class ResultadoRealEvent(BaseModel):
    """Lo que reporta el Gestor al finalizar el secado de un lote (RNF-19): el tiempo real que
    tardó, que sí se conoce en ese momento. calidad_real (el puntaje de catación) YA NO va aquí --
    normalmente no existe todavía cuando el lote se acaba de secar, llega semanas después vía
    CatacionEvent/POST /internal/lotes/{id_lote}/catacion. Este endpoint hace upsert: si ya
    existe una fila de retroalimentación para el lote (porque la catación llegó primero, caso
    raro pero posible), actualiza tiempo_real_horas sin tocar calidad_real."""
    tiempo_real_horas: Optional[float] = Field(
        default=None,
        description="Horas reales totales de secado. Si no se manda, se calcula desde fecha_inicio_secado del lote hasta ahora.",
    )


class ResultadoRealResponse(BaseModel):
    id_retroalimentacion: int
    mensaje: str


class CatacionEvent(BaseModel):
    """Puntaje real de catación (escala SCA, protocolo de la Specialty Coffee Association) que
    llega normalmente semanas después de que el lote terminó de secarse -- ver Documento de
    Calidad del Café, Sección 7, y la discusión de diseño de la migración a esta escala.
    Requiere que el lote ya haya reportado resultado-real (tiempo de secado); si no, 404."""
    puntaje_sca: float = Field(
        ge=0, le=100,
        description="Puntaje de catación 0-100 (protocolo SCA), asignado por un catador/Q Grader",
    )


class CatacionResponse(BaseModel):
    id_retroalimentacion: int
    mensaje: str