# Dockerfile
# microservicioMLL -- FastAPI + uvicorn, un solo proceso. app/services/poller.py corre como
# tarea asyncio de fondo dentro de ESTE mismo proceso (arrancada en el evento startup de
# app/main.py:33), no es un worker/servicio separado -- no hace falta un segundo contenedor.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app/ml/artifacts/isolation_forest.joblib viaja dentro de la imagen: se carga desde disco al
# arrancar (app/services/anomaly_detector.py:27-36), no desde una URL externa.
COPY app/ ./app/
# app/api/routes/internal.py:21 importa ML.monitoreo A NIVEL DE MODULO (se ejecuta en el
# arranque, no solo al llamar /internal/monitoreo/salud); esa cadena de imports necesita
# ML/entrenamiento.py, ML/evaluacion.py y ML/prediccion_lluvia_ga.py o el proceso no levanta.
COPY ML/ ./ML/

EXPOSE 8000

# PORT no existe en app/core/config.py (solo smtp_port) -- se resuelve aqui, a nivel Docker,
# sin tocar la app: ${PORT:-8000} usa la variable de entorno del contenedor si existe, si no
# cae a 8000. Shell form (no exec form) a proposito, para que la expansion ${...} funcione.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
