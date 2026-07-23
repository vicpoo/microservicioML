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
# app/api/routes/nlp.py y app/api/routes/history.py (ambas importadas por main.py en el
# arranque) importan NLP.clasificar_texto / NLP.buscar_reportes / NLP.generar_reporte /
# NLP.recopilar_datos_reporte / NLP.registrar_reporte a nivel de modulo -- sin esta carpeta
# el proceso tampoco levanta.
COPY NLP/ ./NLP/

# scripts/ SOLO hace falta si activas REENTRENAMIENTO_AUTOMATICO_ENABLED=true (ver
# app/services/reentrenador.py): recolectar_datos_reales.py + train_models.py corren dentro de
# este mismo proceso para reentrenar sin depender de que alguien los corra a mano. El import es
# perezoso (dentro de la funcion, no a nivel de modulo), asi que si dejas esto sin copiar y el
# flag sigue en false (default), el servicio arranca igual de bien que antes.
COPY scripts/ ./scripts/
# Carpetas de datos que scripts/recolectar_datos_reales.py y scripts/train_models.py leen/escriben
# (CSVs intermedios) -- se crean solas con os.makedirs si no existen, pero copiar data/raw/ ya
# sembrado evita un primer ciclo con el CSV vacio si ya tenias uno recolectado localmente.
COPY data/ ./data/

EXPOSE 8000

# PORT no existe en app/core/config.py (solo smtp_port) -- se resuelve aqui, a nivel Docker,
# sin tocar la app: ${PORT:-8000} usa la variable de entorno del contenedor si existe, si no
# cae a 8000. Shell form (no exec form) a proposito, para que la expansion ${...} funcione.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
