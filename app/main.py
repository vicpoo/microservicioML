# Archivo: app/main.py
# Carpeta: microservicioMLL/app/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

import asyncio
import logging

from fastapi import FastAPI

from app.api.routes import dispositivos, history, inference, internal, nlp
from app.core.config import get_settings
from app.models.database import init_db
from app.services import poller, reentrenador

logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title="Microservicio MLL", version=settings.modelo_version)
app.include_router(inference.router, prefix="/api/v1")   # api key (manual/testing)
app.include_router(internal.router, prefix="/api/v1")     # api key (Gestor -> MLL)
app.include_router(history.router, prefix="/api/v1")       # api key (opcional: historial)
app.include_router(dispositivos.router, prefix="/api/v1")  # api key (registro de tokens FCM)
app.include_router(nlp.router, prefix="/api/v1")            # api key (clasificador de texto libre)


@app.on_event("startup")
async def startup_event():
    init_db()
    if settings.polling_enabled:
        # Paso 12: arranca el poller como tarea de fondo dentro de este mismo proceso -- no
        # hace falta un worker/servicio aparte para que el MLL sea "tiempo real" (ver
        # app/services/poller.py). Se guarda en app.state para que no se pierda la referencia
        # a la tarea (si no, el garbage collector podría cancelarla).
        app.state.poller_task = asyncio.create_task(poller.loop_polling())
    else:
        logger.info("[main] POLLING_ENABLED=false -- el servicio queda 100%% reactivo (solo webhook).")

    if settings.reentrenamiento_automatico_enabled:
        # Cierra el ciclo de aprendizaje real (ver app/services/reentrenador.py): sin esto, el
        # modelo nunca se actualiza con calidad_real/horas_restantes reales aunque se acumulen,
        # a menos que alguien corra los scripts a mano.
        app.state.reentrenador_task = asyncio.create_task(reentrenador.loop_reentrenamiento())
    else:
        logger.info("[main] REENTRENAMIENTO_AUTOMATICO_ENABLED=false -- el reentrenamiento sigue siendo manual (scripts/train_models.py).")


@app.on_event("shutdown")
async def shutdown_event():
    task = getattr(app.state, "poller_task", None)
    if task is not None:
        task.cancel()
    reentrenador_task = getattr(app.state, "reentrenador_task", None)
    if reentrenador_task is not None:
        reentrenador_task.cancel()


@app.get("/health")
def health():
    return {"status": "ok", "service": "microservicioMLL", "modelo_version": settings.modelo_version}