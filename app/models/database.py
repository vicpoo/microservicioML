#app/models/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_connect_args = {}
if settings.database_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(settings.database_url, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    # Importa todos los modelos para que queden registrados en Base.metadata
    from app.models import (  # noqa: F401
        alertas,
        dispositivos_usuario,
        estado_polling,
        inferencias_ml,
        lecturas_ambientales,
        lotes_cafe,
        modelos_ml,
        notificacion_push_anomalia,
        predicciones,
        recomendaciones,
        reportes_lote,
        retroalimentacion_ml,
        sensores,
    )

    # create_all usa CREATE TABLE IF NOT EXISTS: en Neon (donde las tablas ya existen)
    # no toca nada; en un sqlite de desarrollo/tests las crea desde cero.
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
