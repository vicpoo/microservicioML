# Archivo: tests/test_api.py
# Carpeta: microservicioMLL/tests/

from fastapi.testclient import TestClient

from app.main import app
from app.models.database import SessionLocal, init_db
from app.models.lecturas_ambientales import LecturaAmbiental
from app.models.lotes_cafe import LoteCafe
from app.models.sensores import Sensor

init_db()

USUARIO_A = 1
USUARIO_B = 2


def _seed_lotes():
    db = SessionLocal()
    try:
        if db.query(LoteCafe).filter(LoteCafe.id_lote == 1).first():
            return
        db.add(Sensor(id_sensor=1, mac_address="AA:BB:CC:00:00:01", tipo="ambos", modelo="BMP280+DS18B20"))
        db.add(LoteCafe(
            id_lote=1, id_usuario=USUARIO_A, id_sensor=1, nombre_lote="Lote de A",
            codigo_qr="QR-TEST-1", tipo_proceso="lavado",
        ))
        db.commit()
    finally:
        db.close()


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_anomaly_sin_lote():
    payload = {
        "id_usuario": USUARIO_A,
        "tipo_proceso": "lavado",
        "lecturas": {
            "temperatura_grano": 27.0, "temperatura_ambiental": 25.0,
            "humedad_grano": 30.0, "lluvia": 0.0, "luz": 40000,
        },
    }
    response = client.post("/api/v1/anomalies/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["alerta_generada"] is False


def test_detect_anomaly_critica_con_lote():
    _seed_lotes()
    payload = {
        "id_usuario": USUARIO_A,
        "id_lote": 1,
        "tipo_proceso": "lavado",
        "lecturas": {
            "temperatura_grano": 30.0, "temperatura_ambiental": 26.0,
            "humedad_grano": 40.0, "lluvia": 0.9, "luz": 5000,
        },
    }
    response = client.post("/api/v1/anomalies/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["es_anomalia"] is True
    assert data["nivel_severidad"] == "critico"
    assert data["alerta_generada"] is True  # critico SÍ alerta

    # advertencia leve: NO debe generar alerta (solo riesgo/critico alertan)
    payload_leve = dict(payload, lecturas={
        "temperatura_grano": 36.0, "temperatura_ambiental": 30.0,
        "humedad_grano": 30.0, "lluvia": 0.0, "luz": 40000,
    })
    resp_leve = client.post("/api/v1/anomalies/detect", json=payload_leve)
    assert resp_leve.status_code == 200
    assert resp_leve.json()["nivel_severidad"] == "advertencia"
    assert resp_leve.json()["alerta_generada"] is False


def test_gestor_dispara_pipeline_via_endpoint_interno():
    _seed_lotes()
    db = SessionLocal()
    try:
        db.add(LecturaAmbiental(
            id_sensor=1, id_lote=1, temperatura=26.0, humedad_grano=40,
            temperatura_grano=30.0, luz=5000, lluvia_detectada=True,
        ))
        db.commit()
    finally:
        db.close()

    response = client.post("/api/v1/internal/lecturas/nuevas", json={"id_lote": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["nivel_severidad"] == "critico"
    assert data["alerta_generada"] is True


def test_historial_respeta_dueno_del_lote():
    _seed_lotes()
    hist_ok = client.get("/api/v1/anomalies", params={"id_lote": 1, "id_usuario": USUARIO_A})
    assert hist_ok.status_code == 200
    assert len(hist_ok.json()) >= 1

    preds_ok = client.get("/api/v1/anomalies/1/predicciones", params={"id_usuario": USUARIO_A})
    assert preds_ok.status_code == 200
    assert len(preds_ok.json()) >= 1

    # el lote 1 es de USUARIO_A, no de USUARIO_B
    hist_ajeno = client.get("/api/v1/anomalies", params={"id_lote": 1, "id_usuario": USUARIO_B})
    assert hist_ajeno.status_code == 403


def test_resultado_real_retroalimentacion():
    _seed_lotes()
    db = SessionLocal()
    try:
        db.add(LecturaAmbiental(
            id_sensor=1, id_lote=1, temperatura=25.0, humedad_grano=11,
            temperatura_grano=27.0, luz=30000, lluvia_detectada=False,
        ))
        db.commit()
    finally:
        db.close()

    # calidad_real ya NO se manda aquí (ver CatacionEvent) -- solo tiempo_real_horas, que sí se
    # conoce en el momento de finalizar el lote.
    payload = {"tiempo_real_horas": 180.5}
    response = client.post("/api/v1/internal/lotes/1/resultado-real", json=payload)
    assert response.status_code == 201
    id_retro = response.json()["id_retroalimentacion"]
    assert id_retro > 0

    # Reintento del Gestor (mismo lote, mismo endpoint): debe actualizar la misma fila (upsert
    # por id_lote), no crear una segunda -- id_retroalimentacion no cambia.
    response_retry = client.post("/api/v1/internal/lotes/1/resultado-real", json={"tiempo_real_horas": 181.0})
    assert response_retry.status_code == 201
    assert response_retry.json()["id_retroalimentacion"] == id_retro

    resp_lote_inexistente = client.post("/api/v1/internal/lotes/9999/resultado-real", json=payload)
    assert resp_lote_inexistente.status_code == 404


def test_catacion_requiere_resultado_real_primero():
    _seed_lotes()
    db = SessionLocal()
    try:
        if not db.query(LoteCafe).filter(LoteCafe.id_lote == 2).first():
            db.add(LoteCafe(
                id_lote=2, id_usuario=USUARIO_A, id_sensor=1, nombre_lote="Lote de A (catación)",
                codigo_qr="QR-TEST-2", tipo_proceso="lavado",
            ))
            db.commit()
        db.add(LecturaAmbiental(
            id_sensor=1, id_lote=2, temperatura=25.0, humedad_grano=11,
            temperatura_grano=27.0, luz=30000, lluvia_detectada=False,
        ))
        db.commit()
    finally:
        db.close()

    # Todavía no se reportó resultado-real para el lote 2 -> no hay dónde guardar el puntaje.
    resp_sin_tiempo = client.post("/api/v1/internal/lotes/2/catacion", json={"puntaje_sca": 87.5})
    assert resp_sin_tiempo.status_code == 404

    # Ahora sí se reporta el tiempo real primero (flujo normal: finalizar_lote).
    client.post("/api/v1/internal/lotes/2/resultado-real", json={"tiempo_real_horas": 150.0})

    resp_ok = client.post("/api/v1/internal/lotes/2/catacion", json={"puntaje_sca": 87.5})
    assert resp_ok.status_code == 200

    # Fuera de rango (escala SCA es 0-100) debe rechazarse por validación.
    resp_invalido = client.post("/api/v1/internal/lotes/2/catacion", json={"puntaje_sca": 150})
    assert resp_invalido.status_code == 422
