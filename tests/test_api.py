#tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200


def test_detect_anomaly():
    payload = {
        'id_lote': 1,
        'tipo_proceso': 'lavado',
        'lecturas': {
            'temperatura_grano': 38.5,
            'temperatura_ambiental': 32.0,
            'humedad_ambiental': 85.0,
            'humedad_grano': 45.0,
            'viento': 0.5,
            'lluvia': 0.0,
            'luz': 45000,
        },
    }
    response = client.post('/api/v1/anomalies/detect', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'es_anomalia' in data
    assert 'nivel_severidad' in data
