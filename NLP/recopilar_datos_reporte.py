#NLP/recopilar_datos_reporte.py
"""
NLP/recopilar_datos_reporte.py

Paso 1 del "mini-pipeline" de PLN (Procesamiento de Lenguaje Natural) de kajve: Generación de
Lenguaje Natural (NLG) para reportes legibles de un lote, a partir de datos que el pipeline de
ML (carpeta `ML/`) y el microservicio (`app/`) ya producen -- no hace falta ninguna tabla ni
dato nuevo.

Esta carpeta (`NLP/`) es una propuesta separada de `ML/` a propósito: `ML/` es el pipeline de
Machine Learning "clásico" (detección de anomalías, predicción de tiempo/calidad/lluvia -- ver
`ML/README.md`, pasos 1-12); `NLP/` es la capa de PLN que TOMA la salida de ese pipeline
(alertas, predicciones, recomendaciones) y la convierte en texto en lenguaje natural. Son
técnicas de IA distintas resolviendo problemas distintos, así que viven en carpetas distintas,
aunque ambas sean parte del mismo microservicio.

Responsabilidad de este módulo (paso 1, "qué entra en el reporte"): recolectar y estructurar
los datos crudos de un lote -- SIN redactar ni una palabra de texto todavía. Esa parte
(plantillas de redacción) es el paso 2 (`NLP/generar_reporte.py`, pendiente). Separar
"recolección de datos" de "redacción" es el mismo principio que ya sigue `ML/entrenamiento.py`
vs `ML/evaluacion.py`: cada módulo hace una sola cosa, para poder probarlas por separado.

Fuentes de datos (todas de la BD real, mismos modelos que usa el resto de la app):
  - `lotes_cafe`      -> nombre, tipo de proceso, horas transcurridas.
  - `alertas`         -> cuántas hubo, de qué tipo, de qué severidad, la más reciente.
  - `predicciones`     -> la más reciente: tiempo estimado, calidad estimada, riesgo de lluvia.
  - `recomendaciones` -> los textos activos más recientes (ya redactados por
    `app/services/recommender.py` -- el paso 2 de PLN los reutiliza como insumo, no los repite).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.alertas import Alerta
from app.models.lotes_cafe import LoteCafe
from app.models.predicciones import Prediccion
from app.models.recomendaciones import Recomendacion
from app.services.lectura_utils import calcular_horas_transcurridas

# Cuántas recomendaciones distintas (más recientes) se incluyen en el reporte -- un lote con
# muchas alertas repetidas puede acumular decenas de filas en `recomendaciones` con el mismo
# texto; se listan solo las últimas N, sin duplicados de texto.
MAX_RECOMENDACIONES_REPORTE = 5

# Cuántos mensajes de alerta (texto completo) se traen como corpus candidato para el Nivel 2
# de NLG (BM25, ver NLP/rankear_eventos.py) -- no hace falta el historial completo de un lote
# con cientos de alertas, un conjunto razonable de las más recientes basta.
MAX_MENSAJES_ALERTA = 50


@dataclass
class UltimaAlerta:
    tipo: str
    severidad: str
    mensaje: Optional[str]
    fecha: Optional[datetime]
    atendida: bool


@dataclass
class UltimaPrediccion:
    tiempo_estimado_horas: Optional[float]
    calidad_estimada: Optional[str]
    confianza: Optional[float]
    riesgo_lluvia_proxima: Optional[bool]
    horas_anticipacion_lluvia: Optional[int]
    fecha: Optional[datetime]


@dataclass
class DatosReporteLote:
    """Todo lo que el paso 2 (redacción) necesita para armar el texto de un lote -- estructurado,
    sin ninguna palabra de lenguaje natural todavía."""

    id_lote: int
    nombre_lote: str
    tipo_proceso: str
    estado: Optional[str]
    horas_transcurridas: float
    fecha_inicio_secado: Optional[datetime]
    fecha_fin_secado: Optional[datetime]

    total_alertas: int
    alertas_por_tipo: Dict[str, int]
    alertas_por_severidad: Dict[str, int]
    ultima_alerta: Optional[UltimaAlerta]

    ultima_prediccion: Optional[UltimaPrediccion]

    recomendaciones_activas: List[str] = field(default_factory=list)

    # Mensajes de alerta (texto completo, no solo el tipo) -- insumo del Nivel 2 de NLG
    # (NLP/rankear_eventos.py, BM25): cuando hay muchos, se usan para elegir un resumen
    # extractivo de los más relevantes en vez de listar el conteo nada más. Se recolectan aquí
    # (paso 1) para que ese módulo no tenga que tocar la BD por su cuenta.
    mensajes_alertas: List[str] = field(default_factory=list)

    fecha_generado: datetime = field(default_factory=datetime.utcnow)


def _alertas_por_columna(db: Session, id_lote: int, columna) -> Dict[str, int]:
    filas = (
        db.query(columna, func.count(Alerta.id_alerta))
        .filter(Alerta.id_lote == id_lote)
        .group_by(columna)
        .all()
    )
    return {valor: int(conteo) for valor, conteo in filas}


def _ultima_alerta(db: Session, id_lote: int) -> Optional[UltimaAlerta]:
    alerta = (
        db.query(Alerta)
        .filter(Alerta.id_lote == id_lote)
        .order_by(Alerta.fecha_generada.desc())
        .first()
    )
    if alerta is None:
        return None
    return UltimaAlerta(
        tipo=alerta.tipo_alerta,
        severidad=alerta.nivel_severidad,
        mensaje=alerta.mensaje,
        fecha=alerta.fecha_generada,
        atendida=bool(alerta.atendida),
    )


def _ultima_prediccion(db: Session, id_lote: int) -> Optional[UltimaPrediccion]:
    prediccion = (
        db.query(Prediccion)
        .filter(Prediccion.id_lote == id_lote)
        .order_by(Prediccion.fecha_prediccion.desc())
        .first()
    )
    if prediccion is None:
        return None
    return UltimaPrediccion(
        tiempo_estimado_horas=float(prediccion.tiempo_estimado_horas) if prediccion.tiempo_estimado_horas is not None else None,
        calidad_estimada=prediccion.calidad_estimada,
        confianza=float(prediccion.confianza) if prediccion.confianza is not None else None,
        riesgo_lluvia_proxima=prediccion.riesgo_lluvia_proxima,
        horas_anticipacion_lluvia=prediccion.horas_anticipacion_lluvia,
        fecha=prediccion.fecha_prediccion,
    )


def _recomendaciones_activas(db: Session, id_lote: int, limite: int = MAX_RECOMENDACIONES_REPORTE) -> List[str]:
    filas = (
        db.query(Recomendacion)
        .filter(Recomendacion.id_lote == id_lote)
        .order_by(Recomendacion.fecha_generada.desc())
        .limit(limite * 3)  # de sobra para poder deduplicar por texto y aun así llegar a `limite`
        .all()
    )
    vistos = []
    for r in filas:
        if r.texto not in vistos:
            vistos.append(r.texto)
        if len(vistos) >= limite:
            break
    return vistos


def _mensajes_alertas(db: Session, id_lote: int, limite: int = MAX_MENSAJES_ALERTA) -> List[str]:
    """Texto completo (no solo el tipo) de las alertas más recientes -- corpus de entrada del
    Nivel 2 de NLG (BM25, ver NLP/rankear_eventos.py). Se limita a `limite` para no armar un
    corpus enorme en un lote con cientos de alertas acumuladas; BM25 igual solo necesita un
    conjunto razonable de candidatos entre los que elegir, no el historial completo."""
    filas = (
        db.query(Alerta.mensaje)
        .filter(Alerta.id_lote == id_lote, Alerta.mensaje.isnot(None))
        .order_by(Alerta.fecha_generada.desc())
        .limit(limite)
        .all()
    )
    return [m for (m,) in filas]


def recopilar_datos_lote(db: Session, id_lote: int) -> Optional[DatosReporteLote]:
    """Punto de entrada del paso 1. Devuelve None si el lote no existe (el paso 3, el endpoint,
    decide qué HTTP status devolver en ese caso -- este módulo no sabe nada de HTTP)."""
    lote = db.query(LoteCafe).filter(LoteCafe.id_lote == id_lote).first()
    if lote is None:
        return None

    alertas_por_tipo = _alertas_por_columna(db, id_lote, Alerta.tipo_alerta)
    alertas_por_severidad = _alertas_por_columna(db, id_lote, Alerta.nivel_severidad)
    total_alertas = sum(alertas_por_tipo.values())

    return DatosReporteLote(
        id_lote=lote.id_lote,
        nombre_lote=lote.nombre_lote,
        tipo_proceso=(lote.tipo_proceso or "lavado").lower(),
        estado=lote.estado,
        horas_transcurridas=calcular_horas_transcurridas(lote),
        fecha_inicio_secado=lote.fecha_inicio_secado,
        fecha_fin_secado=lote.fecha_fin_secado,
        total_alertas=total_alertas,
        alertas_por_tipo=alertas_por_tipo,
        alertas_por_severidad=alertas_por_severidad,
        ultima_alerta=_ultima_alerta(db, id_lote),
        ultima_prediccion=_ultima_prediccion(db, id_lote),
        recomendaciones_activas=_recomendaciones_activas(db, id_lote),
        mensajes_alertas=_mensajes_alertas(db, id_lote),
    )
