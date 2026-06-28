# Documentación del microservicio MLL

Este proyecto implementa un microservicio en FastAPI para detectar anomalías en el proceso de secado del café mediante un modelo local de Machine Learning no supervisado (Isolation Forest).

## 1. ¿Qué hace el proyecto?

El microservicio recibe lecturas del proceso de secado, realiza un preprocesamiento básico, ejecuta el modelo para detectar patrones atípicos y guarda cada inferencia en una base de datos local. Además, expone un endpoint para consultar el historial de inferencias.

## 2. Estructura del proyecto

```text
microservicioMLL/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes/
│   │       ├── inference.py
│   │       └── history.py
│   ├── core/
│   │   └── config.py
│   ├── models/
│   │   ├── database.py
│   │   └── inference_record.py
│   ├── schemas/
│   │   ├── inference_request.py
│   │   └── inference_response.py
│   ├── services/
│   │   ├── preprocessor.py
│   │   └── anomaly_detector.py
│   └── ml/
│       └── artifacts/
├── tests/
│   └── test_api.py
├── requirements.txt
├── README.md
└── DOCUMENTACION.md
```

## 3. Explicación de cada archivo

### [app/main.py](app/main.py)
Este es el punto de entrada de la aplicación FastAPI.

¿Qué hace?
- Crea la instancia principal de la app.
- Incluye los routers de inferencia e historial.
- Inicia la base de datos al arrancar la aplicación.
- Expone el endpoint /health para verificar que el servicio está arriba.

### [app/api/routes/inference.py](app/api/routes/inference.py)
Este archivo define el endpoint principal para realizar inferencias.

¿Qué hace?
- Recibe un JSON con las lecturas del proceso.
- Preprocesa los datos.
- Ejecuta el modelo de detección de anomalías.
- Devuelve un JSON con el resultado.
- Guarda la inferencia en la base de datos.

### [app/api/routes/history.py](app/api/routes/history.py)
Este archivo define el endpoint para consultar el historial de inferencias.

¿Qué hace?
- Recupera las inferencias guardadas en la base de datos.
- Permite filtrar por id_lote.
- Soporta paginación con limit y offset.

### [app/core/config.py](app/core/config.py)
Define la configuración general de la aplicación.

¿Qué hace?
- Establece la URL de la base de datos.
- Permite leer configuraciones desde un archivo .env si se desea.

### [app/models/database.py](app/models/database.py)
Gestiona la conexión y la creación de la base de datos.

¿Qué hace?
- Crea el motor de SQLAlchemy.
- Define la sesión de base de datos.
- Crea las tablas al iniciar la app.

### [app/models/inference_record.py](app/models/inference_record.py)
Define el modelo de la tabla que guarda las inferencias.

¿Qué hace?
- Representa la tabla inferencias_anomalias.
- Guarda información como:
  - id_lote
  - tipo_proceso
  - payload_entrada
  - es_anomalia
  - score_anomalia
  - nivel_severidad
  - mensaje
  - modelo_version
  - fecha_inferencia

### [app/schemas/inference_request.py](app/schemas/inference_request.py)
Define el esquema de entrada del endpoint de inferencia.

¿Qué hace?
- Valida la estructura del JSON que envía el cliente.
- Define los campos esperados, como id_lote, tipo_proceso, timestamp y lecturas.

### [app/schemas/inference_response.py](app/schemas/inference_response.py)
Define la estructura de respuesta del endpoint.

¿Qué hace?
- Estandariza la respuesta JSON que devuelve la API.
- Asegura que el cliente reciba campos claros y consistentes.

### [app/services/preprocessor.py](app/services/preprocessor.py)
Se encarga de transformar las lecturas antes de pasarlas al modelo.

¿Qué hace?
- Toma las variables de entrada.
- Genera características derivadas como:
  - delta_temp
  - indice_moho
  - lluvia_binaria
- Prepara los datos en un formato útil para el modelo.

### [app/services/anomaly_detector.py](app/services/anomaly_detector.py)
Implementa la lógica del modelo de detección.

¿Qué hace?
- Carga o entrena un modelo local de Isolation Forest.
- Recibe los datos transformados.
- Devuelve si la muestra es anómala, el score y las variables contribuyentes.

### [app/ml/artifacts/](app/ml/artifacts/)
Carpeta donde se guardan los artefactos del modelo.

¿Qué hace?
- Almacena el modelo entrenado en formato .joblib.
- Permite cargar el modelo sin entrenarlo cada vez que se inicia la app.

### [tests/test_api.py](tests/test_api.py)
Contiene pruebas básicas para verificar el funcionamiento del servicio.

¿Qué hace?
- Prueba el endpoint de health.
- Prueba el endpoint de inferencia.

### [requirements.txt](requirements.txt)
Lista las dependencias del proyecto.

¿Qué hace?
- Permite instalar todo lo necesario para correr FastAPI, SQLAlchemy, scikit-learn y pytest.

### [README.md](README.md)
Documento breve para iniciar el proyecto.

## 4. Flujo de funcionamiento

1. El cliente envía un JSON al endpoint /api/v1/anomalies/detect.
2. La API valida el payload.
3. El preprocesador transforma los datos.
4. El modelo detecta si la muestra es anómala.
5. La API devuelve el resultado en JSON.
6. La inferencia se guarda en la base de datos.
7. El historial se puede consultar desde /api/v1/anomalies.

## 5. Cómo probar el microservicio

### Opción 1: Ejecutar la app localmente

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Inicia la aplicación:

```bash
uvicorn app.main:app --reload
```

La app quedará disponible en:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

### Opción 2: Probar con curl

#### Verificar que el servicio está vivo

```bash
curl http://127.0.0.1:8000/health
```

#### Hacer una inferencia

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/anomalies/detect" -H "Content-Type: application/json" -d "{
  \"id_lote\": 1,
  \"tipo_proceso\": \"lavado\",
  \"lecturas\": {
    \"temperatura_grano\": 38.5,
    \"temperatura_ambiental\": 32.0,
    \"humedad_ambiental\": 85.0,
    \"humedad_grano\": 45.0,
    \"viento\": 0.5,
    \"lluvia\": 0.0,
    \"luz\": 45000
  }
}"
```

#### Consultar historial

```bash
curl http://127.0.0.1:8000/api/v1/anomalies
```

### Opción 3: Probar desde Swagger UI

1. Abre la URL:
   - http://127.0.0.1:8000/docs
2. Busca el endpoint /api/v1/anomalies/detect.
3. Haz clic en "Try it out".
4. Envía un ejemplo como este:

```json
{
  "id_lote": 1,
  "tipo_proceso": "lavado",
  "lecturas": {
    "temperatura_grano": 38.5,
    "temperatura_ambiental": 32.0,
    "humedad_ambiental": 85.0,
    "humedad_grano": 45.0,
    "viento": 0.5,
    "lluvia": 0.0,
    "luz": 45000
  }
}
```

## 6. Cómo ejecutar las pruebas

```bash
pytest -q
```

## 7. Notas importantes

- La base de datos actual es local con SQLite.
- El modelo es una primera versión simple y sirve como base para el proyecto final.
- Puedes ir mejorando el preprocesamiento y el modelo con tus propios datos reales.

## 8. Siguiente paso recomendado

Para el proyecto final, lo siguiente sería:
- usar tus datos de [data/raw](data/raw) para entrenar mejor el modelo,
- agregar validaciones más completas,
- mejorar la lógica de severidad,
- y documentar los resultados en el notebook y el PDF para la entrega.
