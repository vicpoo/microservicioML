from fastapi import FastAPI
from app.api.routes import inference, history
from app.core.config import get_settings
from app.models.database import init_db

settings = get_settings()
app = FastAPI(title="Microservicio MLL", version="0.1.0")

app.include_router(inference.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok", "service": "microservicioMLL"}
