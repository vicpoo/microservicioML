# Microservicio MLL

Este proyecto implementa una primera versión de un microservicio FastAPI para detectar anomalías de secado de café mediante un modelo local de Isolation Forest.

## Ejecutar localmente

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

- GET /health
- POST /api/v1/anomalies/detect
- GET /api/v1/anomalies
- Swagger UI en /docs
