# Archivo: app/main.py
# Carpeta: microservicioMLL/app/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from fastapi import FastAPI

from app.api.routes import history, inference, internal
from app.core.config import get_settings
from app.models.database import init_db

settings = get_settings()
app = FastAPI(title="Microservicio MLL", version=settings.modelo_version)
app.include_router(inference.router, prefix="/api/v1")   # api key (manual/testing)
app.include_router(internal.router, prefix="/api/v1")     # api key (Gestor -> MLL)
app.include_router(history.router, prefix="/api/v1")       # api key (opcional: historial)


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "service": "microservicioMLL", "modelo_version": settings.modelo_version}