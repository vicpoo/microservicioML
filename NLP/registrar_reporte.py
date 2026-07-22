#NLP/registrar_reporte.py
"""
NLP/registrar_reporte.py

Paso 4 del mini-pipeline de PLN: persistencia del historial de reportes. Los pasos 1-3
(recolectar datos -> redactar texto -> exponerlo por API) generaban el reporte al vuelo en cada
llamada, sin guardar nada. Este módulo agrega una fila a `reportes_lote` (tabla nueva, sección 8
de `migration.sql`) cada vez que se genera un reporte -- mismo criterio que ya siguen
`predicciones`/`alertas`/`recomendaciones` en `app/services/notifier.py`: se acumula historial,
no se sobrescribe la fila anterior, así se puede ver cómo cambió el reporte de un lote con el
tiempo (por ejemplo, comparar el reporte de hace 3 días contra el de ahora).

Separado de `generar_reporte.py` a propósito: ese módulo es puro texto (no toca la BD); este
módulo es puro I/O de BD (no redacta nada) -- mismo principio de una responsabilidad por módulo
que ya sigue el resto de `NLP/` y `ML/`.
"""
from typing import List, TypedDict

from sqlalchemy.orm import Session

from app.models.reportes_lote import ReporteLote


class ReporteHistorico(TypedDict):
    id_reporte: int
    id_lote: int
    reporte_texto: str
    fecha_generado: str


def guardar_reporte(db: Session, id_lote: int, reporte_texto: str) -> int:
    registro = ReporteLote(id_lote=id_lote, reporte_texto=reporte_texto)
    db.add(registro)
    db.commit()
    db.refresh(registro)
    return registro.id_reporte


def historial_reportes(db: Session, id_lote: int, limit: int = 10) -> List[ReporteHistorico]:
    registros = (
        db.query(ReporteLote)
        .filter(ReporteLote.id_lote == id_lote)
        # Desempate por id_reporte: fecha_generado tiene resolución de 1 segundo (server_default
        # now()) y dos reportes pedidos casi seguido (típico en pruebas automatizadas, y posible
        # en producción con clientes rápidos) pueden empatar en el mismo segundo -- sin este
        # desempate, el orden entre ellos queda indefinido en vez de "más reciente primero".
        .order_by(ReporteLote.fecha_generado.desc(), ReporteLote.id_reporte.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id_reporte": r.id_reporte,
            "id_lote": r.id_lote,
            "reporte_texto": r.reporte_texto,
            "fecha_generado": r.fecha_generado.isoformat() if r.fecha_generado else None,
        }
        for r in registros
    ]
