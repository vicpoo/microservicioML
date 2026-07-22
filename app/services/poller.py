#app/services/poller.py
"""
Hace que el microservicio sea un servicio "en tiempo real" de verdad, no solo reactivo. El
disparador normal sigue siendo que el Gestor llame POST /internal/lecturas/nuevas apenas
inserta una lectura -- eso es instantáneo y es el camino preferido. Este módulo es la RED DE
SEGURIDAD: revisa lecturas_ambientales cada POLLING_INTERVALO_SEGUNDOS por si algo se coló sin
avisar (Gestor caído, error de red, un reinicio a medio camino), y lo procesa solo. Así el
servicio detecta datos nuevos aunque el webhook nunca se haya llamado.

Cursor compartido: tanto este poller como POST /internal/lecturas/nuevas avanzan la MISMA fila
(tabla ml_estado_polling, ver migration.sql sección 6) después de procesar una lectura. Sin
esto, cada lectura que el Gestor sí avisó a tiempo se volvería a procesar cuando le tocara el
turno al poller, duplicando predicciones/alertas/notificaciones push para el mismo dato.

Se arranca como una tarea de fondo (asyncio) en el startup de FastAPI -- ver app/main.py. No
existe como proceso separado a propósito: así no hace falta un segundo servicio/worker
desplegado aparte para tener "tiempo real", corre dentro del mismo proceso del API.
"""
import asyncio
import logging

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.database import SessionLocal
from app.models.estado_polling import EstadoPolling
from app.models.lecturas_ambientales import LecturaAmbiental
from app.models.lotes_cafe import LoteCafe
from app.services.lectura_utils import calcular_horas_transcurridas, construir_features

logger = logging.getLogger(__name__)
settings = get_settings()

_ESTADO_ID = 1  # fila única (singleton) -- un solo cursor por servicio, no "uno por lote"


def _obtener_o_crear_estado(db: Session) -> EstadoPolling:
    estado = db.query(EstadoPolling).filter(EstadoPolling.id == _ESTADO_ID).first()
    if estado is None:
        estado = EstadoPolling(id=_ESTADO_ID, ultima_id_lectura_procesada=0)
        db.add(estado)
        db.commit()
        db.refresh(estado)
    return estado


def marcar_procesada(db: Session, id_lectura: int) -> None:
    """Avanza el cursor compartido. La llama tanto este módulo (después de procesar en su
    propio ciclo) como POST /internal/lecturas/nuevas (después de procesar el webhook), para
    que el otro camino sepa que esa lectura ya no hace falta reprocesarla."""
    estado = _obtener_o_crear_estado(db)
    if id_lectura > estado.ultima_id_lectura_procesada:
        estado.ultima_id_lectura_procesada = id_lectura
        estado.actualizado_en = datetime.now(timezone.utc)
        db.commit()


def procesar_lecturas_nuevas(db: Session, batch_size: int = None) -> int:
    """Un solo ciclo: busca lecturas con id_lectura > cursor, las procesa en orden y avanza
    el cursor una por una (si algo falla a medio lote, no se pierde el progreso ya hecho).
    Devuelve cuántas se procesaron con éxito."""
    # Import perezoso: ejecutar_pipeline vive en app.api.routes.inference, que a su vez podría
    # (en el futuro) querer importar cosas de app.services -- evita un ciclo de imports al
    # nivel de módulo.
    from app.api.routes.inference import ejecutar_pipeline

    batch_size = batch_size or settings.polling_batch_size
    estado = _obtener_o_crear_estado(db)

    nuevas = (
        db.query(LecturaAmbiental)
        .filter(LecturaAmbiental.id_lectura > estado.ultima_id_lectura_procesada)
        .order_by(LecturaAmbiental.id_lectura.asc())
        .limit(batch_size)
        .all()
    )

    procesadas = 0
    for lectura in nuevas:
        try:
            lote = db.query(LoteCafe).filter(LoteCafe.id_lote == lectura.id_lote).first()
            if lote is None:
                # Lectura huérfana (id_lote no existe en lotes_cafe) -- no hay a quién
                # avisarle ni dueño a quién atribuírsela; se salta y se avanza el cursor
                # igual, para no atascar el poller reintentándola cada ciclo para siempre.
                logger.warning(f"[poller] lectura {lectura.id_lectura} referencia el lote "
                                f"{lectura.id_lote}, que no existe en lotes_cafe; se salta.")
                marcar_procesada(db, lectura.id_lectura)
                procesadas += 1
                continue

            tipo_proceso = (lote.tipo_proceso or "lavado").lower()
            features = construir_features(lectura)
            horas_transcurridas = calcular_horas_transcurridas(lote)
            presion_hpa = float(lectura.presion_hpa) if lectura.presion_hpa is not None else None

            ejecutar_pipeline(
                db, lote, lectura.id_lote, tipo_proceso, lectura.id_sensor, features,
                horas_transcurridas, guardar_lectura=False, presion_hpa=presion_hpa,
            )
            marcar_procesada(db, lectura.id_lectura)
            procesadas += 1
        except Exception as exc:
            # No se sigue con el resto del batch en este ciclo: si el error es transitorio
            # (ej. Neon caída un instante) mejor esperar al siguiente ciclo completo que
            # insistir en bucle. El cursor no avanzó para esta lectura, así que se reintenta
            # sola la próxima vez.
            logger.error(f"[poller] error procesando lectura {lectura.id_lectura} "
                          f"(lote {lectura.id_lote}): {exc}")
            db.rollback()
            break

    return procesadas


async def loop_polling() -> None:
    """Tarea de fondo lanzada en el startup de FastAPI. Corre mientras el proceso viva; cada
    ciclo abre su propia sesión de BD y corre la consulta síncrona de SQLAlchemy en un hilo
    aparte (asyncio.to_thread) para no bloquear el event loop del servidor -- si no, cada
    revisión pausaría todas las demás requests que estuvieran llegando en ese momento."""
    logger.info(f"[poller] iniciado -- reviso lecturas_ambientales cada "
                f"{settings.polling_intervalo_segundos}s (batch={settings.polling_batch_size})")
    while True:
        try:
            db = SessionLocal()
            try:
                procesadas = await asyncio.to_thread(procesar_lecturas_nuevas, db)
                if procesadas:
                    logger.info(f"[poller] {procesadas} lectura(s) nueva(s) procesada(s)")
            finally:
                db.close()
        except Exception as exc:  # pragma: no cover -- red/BD caída, no debe tumbar el servicio
            logger.error(f"[poller] error en el ciclo de polling: {exc}")
        await asyncio.sleep(settings.polling_intervalo_segundos)
