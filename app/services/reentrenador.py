#app/services/reentrenador.py
"""
Reentrenamiento automático periódico -- cierra el ciclo de aprendizaje real: hasta antes de
este módulo, que el modelo mejorara con calidad_real/calidad_estimada (o con horas_restantes)
dependía 100% de que alguien se acordara de correr a mano
scripts/recolectar_datos_reales.py + scripts/train_models.py. Si nadie lo corría, daba igual
cuántos lotes reales se acumularan: el modelo desplegado seguía siendo el mismo de siempre.

Mismo patrón que app/services/poller.py (ver también el comentario en el Dockerfile: un solo
proceso uvicorn, sin necesidad de un segundo contenedor, worker, ni cron externo):
- Se arranca como tarea de fondo (asyncio) en el startup de FastAPI -- ver app/main.py.
- Cada ciclo corre en un hilo aparte (asyncio.to_thread) para no bloquear el event loop del
  servidor mientras entrena.
- Apagado por default (REENTRENAMIENTO_AUTOMATICO_ENABLED=false): entrenar consume CPU real
  durante varios segundos/minutos (varios RandomForest + GroupShuffleSplit), y aunque no bloquea
  el loop, sí compite por CPU con las peticiones en vivo del mismo proceso -- debe activarse a
  propósito, igual que FCM_ENABLED/POLLING_ENABLED.

No reentrena ciegamente cada vez que corre el temporizador: antes de tocar nada, reutiliza
ML/monitoreo.py::necesita_reentrenamiento() (la misma lógica que ya expone
GET /internal/monitoreo/salud) para decidir si de verdad hay datos nuevos suficientes. Si no,
solo lo registra en el log y espera al siguiente ciclo.
"""
import asyncio
import logging

from app.core.config import get_settings
from app.models.database import SessionLocal
from ML import monitoreo

logger = logging.getLogger(__name__)
settings = get_settings()


def _reentrenar_si_hace_falta() -> dict:
    """Síncrono a propósito -- se llama vía asyncio.to_thread desde el loop de fondo (ver
    loop_reentrenamiento, abajo). No se llama directo desde una ruta async.

    Import perezoso de scripts.* (no a nivel de módulo): a diferencia de ML/, la carpeta
    scripts/ no viaja dentro de la imagen Docker por default (ver Dockerfile) -- si alguien
    activa REENTRENAMIENTO_AUTOMATICO_ENABLED sin haber agregado `COPY scripts/ ./scripts/`,
    el error debe aparecer AQUÍ (un ciclo fallido, logueado, se reintenta el siguiente) y no
    tumbar el arranque de todo el servicio con un ImportError a nivel de módulo."""
    from scripts import recolectar_datos_reales, train_models

    db = SessionLocal()
    try:
        diagnostico = monitoreo.necesita_reentrenamiento(db)
    finally:
        db.close()

    if not diagnostico.get("necesita_reentrenamiento"):
        logger.info("[reentrenador] sin datos nuevos suficientes todavía, se omite este ciclo")
        return {"reentrenado": False, "razones": diagnostico.get("razones", [])}

    for razon in diagnostico.get("razones", []):
        logger.info(f"[reentrenador] {razon}")
    logger.info("[reentrenador] recolectando datos reales actualizados...")
    recolectar_datos_reales.main()
    logger.info("[reentrenador] reentrenando los 4 artefactos con el dataset actualizado...")
    train_models.main()
    logger.info("[reentrenador] reentrenamiento completado, artefactos actualizados en app/ml/artifacts/")
    return {"reentrenado": True, "razones": diagnostico.get("razones", [])}


async def loop_reentrenamiento() -> None:
    """Tarea de fondo lanzada en el startup de FastAPI (si REENTRENAMIENTO_AUTOMATICO_ENABLED).
    Corre mientras el proceso viva; nunca lanza una excepción hacia afuera -- un fallo en un
    ciclo (ej. Neon caída, CSV corrupto) no debe tumbar el servicio, se reintenta en el
    siguiente ciclo."""
    intervalo_segundos = settings.reentrenamiento_intervalo_horas * 3600
    logger.info(f"[reentrenador] iniciado -- reviso cada {settings.reentrenamiento_intervalo_horas}h "
                "si hay datos nuevos suficientes para reentrenar")
    while True:
        await asyncio.sleep(intervalo_segundos)
        try:
            await asyncio.to_thread(_reentrenar_si_hace_falta)
        except Exception as exc:  # pragma: no cover -- red/BD caída, no debe tumbar el servicio
            logger.error(f"[reentrenador] error durante el ciclo de reentrenamiento (se reintenta "
                         f"en el próximo ciclo): {exc}")
