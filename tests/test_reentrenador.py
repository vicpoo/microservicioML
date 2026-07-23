# Archivo: tests/test_reentrenador.py
# Carpeta: microservicioMLL/tests/
"""Verificación end-to-end (local, SQLite) del ciclo completo de la migración a escala SCA:

    HTTP resultado-real -> HTTP catacion -> retroalimentacion_ml (tiempo + calidad numérica)
    -> necesita_reentrenamiento() detecta datos suficientes
    -> recolectar_datos_reales + train_models reentrenan el regresor de calidad
    -> Predictor carga el artefacto nuevo y devuelve un puntaje 0-100
    -> el reporte NLG (HTTP) menciona ese puntaje en texto

No se puede correr esto contra Neon real desde este entorno (sin acceso a internet); estas
pruebas reproducen el mismo flujo completo con una BD local (SQLite) para confirmar que toda la
plomería de la migración queda bien conectada de punta a punta, sin necesitar la BD de
producción.

IMPORTANTE: igual que test_ml_calidad.py, todo entrenamiento aquí redirige ARTIFACTS_DIR/
REAL_CSV a rutas temporales -- nunca debe tocar app/ml/artifacts/ ni data/raw/ reales."""
import os
from datetime import datetime, timedelta

import scripts.recolectar_datos_reales as recolectar_datos_reales
import scripts.train_models as train_models
from fastapi.testclient import TestClient
from ML import monitoreo

from app.main import app
from app.models.database import SessionLocal, init_db
from app.models.lecturas_ambientales import LecturaAmbiental
from app.models.lotes_cafe import LoteCafe
from app.models.predicciones import Prediccion
from app.models.retroalimentacion_ml import RetroalimentacionML
from app.models.sensores import Sensor
from app.services.predictor import Predictor

init_db()
client = TestClient(app)

ID_USUARIO_E2E = 555001
BASE_ID_LOTE = 555100


def _sembrar_lotes_finalizados_con_retroalimentacion(n=5):
    """Crea n lotes finalizados, cada uno con lecturas ambientales y una fila de
    retroalimentacion_ml (tiempo_real_horas + calidad_real numérica, ya variada entre lotes, no
    constante), suficiente para que necesita_reentrenamiento() detecte que ya se puede entrenar
    el regresor de calidad (MIN_LOTES_CALIDAD = 5)."""
    db = SessionLocal()
    try:
        if not db.query(Sensor).filter(Sensor.id_sensor == 555001).first():
            db.add(Sensor(id_sensor=555001, mac_address="E2E:TEST:SENSOR:1", tipo="ambos", modelo="test"))
            db.commit()

        ids_lote = []
        for i in range(n):
            id_lote = BASE_ID_LOTE + i
            ids_lote.append(id_lote)
            if db.query(LoteCafe).filter(LoteCafe.id_lote == id_lote).first():
                continue
            inicio = datetime.utcnow() - timedelta(days=10)
            fin = datetime.utcnow() - timedelta(days=2)
            db.add(LoteCafe(
                id_lote=id_lote, id_usuario=ID_USUARIO_E2E, id_sensor=555001,
                nombre_lote=f"Lote E2E {i}", codigo_qr=f"QR-E2E-{id_lote}",
                tipo_proceso=["lavado", "honey", "natural"][i % 3],
                estado="finalizado", fecha_inicio_secado=inicio, fecha_fin_secado=fin,
            ))
            for h in range(4):
                db.add(LecturaAmbiental(
                    id_sensor=555001, id_lote=id_lote,
                    temperatura=24.0 + i, temperatura_grano=26.0 + i,
                    humedad_grano=30 - i, luz=30000, lluvia_detectada=False,
                    timestamp=inicio + timedelta(hours=h * 12),
                ))
            db.add(RetroalimentacionML(
                id_lote=id_lote, tipo_proceso=["lavado", "honey", "natural"][i % 3],
                temperatura_grano=26.0 + i, temperatura_ambiental=24.0 + i,
                humedad_grano=11, lluvia_detectada=False, luz=30000,
                tiempo_real_horas=180.0 + i * 5,
                calidad_real=60.0 + i * 8,  # 60, 68, 76, 84, 92 -- variado, no constante
            ))
        db.commit()
        return ids_lote
    finally:
        db.close()


def test_endpoints_resultado_real_y_catacion_alimentan_la_misma_fila():
    """Fase 2 contra la app real (no solo unit tests aislados de internal.py): el flujo HTTP
    completo resultado-real -> catacion debe dejar tiempo_real_horas y calidad_real en la MISMA
    fila de retroalimentacion_ml (upsert por id_lote), no en dos filas separadas."""
    db = SessionLocal()
    try:
        if not db.query(Sensor).filter(Sensor.id_sensor == 555002).first():
            db.add(Sensor(id_sensor=555002, mac_address="E2E:TEST:SENSOR:2", tipo="ambos", modelo="test"))
        id_lote = BASE_ID_LOTE + 900
        if not db.query(LoteCafe).filter(LoteCafe.id_lote == id_lote).first():
            db.add(LoteCafe(
                id_lote=id_lote, id_usuario=ID_USUARIO_E2E, id_sensor=555002,
                nombre_lote="Lote E2E endpoint", codigo_qr=f"QR-E2E-{id_lote}",
                tipo_proceso="lavado",
            ))
        db.commit()
        db.add(LecturaAmbiental(
            id_sensor=555002, id_lote=id_lote, temperatura=25.0, temperatura_grano=27.0,
            humedad_grano=11, luz=30000, lluvia_detectada=False,
        ))
        db.commit()
    finally:
        db.close()

    resp1 = client.post(f"/api/v1/internal/lotes/{id_lote}/resultado-real", json={"tiempo_real_horas": 190.0})
    assert resp1.status_code == 201
    id_retro_1 = resp1.json()["id_retroalimentacion"]

    resp2 = client.post(f"/api/v1/internal/lotes/{id_lote}/catacion", json={"puntaje_sca": 91.0})
    assert resp2.status_code == 200
    assert resp2.json()["id_retroalimentacion"] == id_retro_1  # misma fila, no una nueva

    db = SessionLocal()
    try:
        fila = db.query(RetroalimentacionML).filter(RetroalimentacionML.id_lote == id_lote).first()
        assert fila is not None
        assert float(fila.tiempo_real_horas) == 190.0
        assert float(fila.calidad_real) == 91.0
    finally:
        db.close()


def test_ciclo_completo_reentrenamiento_y_prediccion_numerica(tmp_path, monkeypatch):
    """El E2E principal de la Fase 8: siembra >= MIN_LOTES_CALIDAD lotes con retroalimentación
    real, confirma que necesita_reentrenamiento() lo detecta, corre el ciclo de recolección +
    entrenamiento completo (redirigido a un directorio temporal), y confirma que Predictor
    puede cargar el artefacto resultante y producir un puntaje 0-100 utilizable."""
    _sembrar_lotes_finalizados_con_retroalimentacion(n=5)

    db = SessionLocal()
    try:
        diagnostico = monitoreo.necesita_reentrenamiento(db)
    finally:
        db.close()
    assert diagnostico["necesita_reentrenamiento"] is True
    assert diagnostico["disponibilidad_datos"]["lotes_faltantes_para_calidad"] == 0

    # Redirige TODO el entrenamiento a rutas temporales -- nunca tocar app/ml/artifacts/ ni
    # data/raw/ reales con este dataset de prueba.
    os.makedirs(str(tmp_path), exist_ok=True)
    csv_temporal = str(tmp_path / "lecturas_reales_entrenamiento.csv")
    monkeypatch.setattr(train_models, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(train_models, "REAL_CSV", csv_temporal)
    monkeypatch.setattr(train_models, "PROCESSED_CSV", str(tmp_path / "lecturas_limpias.csv"))

    df = recolectar_datos_reales.construir_dataset()
    assert df["id_lote"].nunique() >= 5
    df.to_csv(csv_temporal, index=False)

    df_limpio = train_models.cargar_y_limpiar()
    resultado_calidad = train_models.entrenar_regresor_calidad(df_limpio)
    assert "omitido" not in resultado_calidad
    assert resultado_calidad["n_lotes"] >= 5
    assert "rmse_puntos" in resultado_calidad

    predictor = Predictor(artifacts_dir=str(tmp_path))
    assert predictor.rf_calidad is not None
    resultado = predictor.predecir(
        tipo_proceso="lavado",
        features={"temperatura_grano": 27.0, "temperatura_ambiental": 24.0, "humedad_grano": 30.0,
                  "lluvia": 0.0, "luz": 31000, "delta_temp": 3.0},
        horas_transcurridas=100.0,
    )
    assert resultado["calidad_estimada"] is not None
    assert 0.0 <= resultado["calidad_estimada"] <= 100.0
    assert resultado["confianza"] is not None
    assert 0.0 <= resultado["confianza"] <= 100.0


def test_reporte_nlg_menciona_puntaje_numerico_de_calidad_via_http():
    """Cierra el lazo hacia el usuario final: si predicciones.calidad_estimada tiene un puntaje
    numérico guardado, el reporte en lenguaje natural (endpoint HTTP real, no solo la función
    generar_reporte_lote de forma aislada) debe mencionarlo como 'NN/100', no como una
    categoría."""
    id_lote = BASE_ID_LOTE  # sembrado por _sembrar_lotes_finalizados_con_retroalimentacion
    db = SessionLocal()
    try:
        if not db.query(LoteCafe).filter(LoteCafe.id_lote == id_lote).first():
            _sembrar_lotes_finalizados_con_retroalimentacion(n=1)
        db.add(Prediccion(
            id_lote=id_lote, id_modelo=1,
            tiempo_estimado_horas=48.0, calidad_estimada=82.0, confianza=75.0,
        ))
        db.commit()
    finally:
        db.close()

    resp = client.get(f"/api/v1/anomalies/{id_lote}/reporte", params={"id_usuario": ID_USUARIO_E2E})
    assert resp.status_code == 200
    texto = resp.json()["reporte_texto"]
    assert "82/100" in texto
    assert "escala SCA" in texto
