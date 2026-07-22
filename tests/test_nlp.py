# Archivo: tests/test_nlp.py
# Carpeta: microservicioMLL/tests/
#
# Paso 5 del mini-pipeline de PLN (NLP/, ver NLP/README.md): pruebas formales que quedan en el
# repo y corren con el resto de la suite (`pytest tests/ -q`), a diferencia de los scripts de
# humo sueltos usados mientras se construía cada paso. Dos niveles:
#   - Unitarias sobre NLP/generar_reporte.py con datos sintéticos (sin BD, rápidas, cubren los
#     casos borde de la redacción: sin alertas, sin predicción, lote finalizado, riesgo de
#     lluvia en ambos sentidos).
#   - De integración sobre el endpoint (con BD sqlite de prueba), igual que tests/test_api.py:
#     aislamiento por dueño, y que el historial de reportes se acumule correctamente.

import tempfile
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.models.database import Base, SessionLocal, init_db
from app.models.alertas import Alerta
from app.models.lotes_cafe import LoteCafe
from app.models.sensores import Sensor
from app.models.reportes_lote import ReporteLote
from NLP.buscar_reportes import buscar_reportes
from NLP.clasificar_texto import ClasificadorTexto
from NLP.entrenar_clasificador_texto import entrenar
from NLP.generar_reporte import generar_reporte_lote
from NLP.preparar_datos_clasificador import MIN_EJEMPLOS_ENTRENAMIENTO, hay_suficientes_datos, recolectar_ejemplos
from NLP.recopilar_datos_reporte import DatosReporteLote, UltimaAlerta, UltimaPrediccion

init_db()

# Rango de IDs alto y separado del que usa tests/test_api.py (lotes 1-2), para poder correr
# ambos archivos en la misma sesión de pytest sin pisarse.
USUARIO_NLP = 9001
USUARIO_OTRO = 9002
ID_LOTE_NLP = 9001

client = TestClient(app)


def _seed_lote_nlp():
    db = SessionLocal()
    try:
        if db.query(LoteCafe).filter(LoteCafe.id_lote == ID_LOTE_NLP).first():
            return
        db.add(Sensor(id_sensor=9001, mac_address="AA:BB:CC:00:09:01", tipo="ambos", modelo="BMP280+DS18B20"))
        db.add(LoteCafe(
            id_lote=ID_LOTE_NLP, id_usuario=USUARIO_NLP, id_sensor=9001,
            nombre_lote="Lote pruebas NLP", codigo_qr="QR-NLP-TEST", tipo_proceso="lavado",
            estado="en_proceso",
            fecha_inicio_secado=(datetime.now(timezone.utc) - timedelta(hours=10)).replace(tzinfo=None),
        ))
        db.commit()
    finally:
        db.close()


# --- Unitarias: NLP/generar_reporte.py directo, sin BD -----------------------------------------

def _datos_base(**overrides) -> DatosReporteLote:
    base = dict(
        id_lote=1, nombre_lote="Lote prueba", tipo_proceso="lavado", estado="en_proceso",
        horas_transcurridas=10.5, fecha_inicio_secado=None, fecha_fin_secado=None,
        total_alertas=0, alertas_por_tipo={}, alertas_por_severidad={},
        ultima_alerta=None, ultima_prediccion=None, recomendaciones_activas=[],
    )
    base.update(overrides)
    return DatosReporteLote(**base)


def test_reporte_sin_historial_no_truena_y_menciona_falta_de_datos():
    texto = generar_reporte_lote(_datos_base())
    assert "No se han registrado alertas" in texto
    assert "No hay suficientes datos históricos todavía" in texto


def test_reporte_con_alertas_cuenta_y_nombra_tipos():
    datos = _datos_base(
        total_alertas=3,
        alertas_por_tipo={"temperatura_alta": 2, "lluvia_detectada": 1},
        alertas_por_severidad={"alta": 2, "critica": 1},
        ultima_alerta=UltimaAlerta(tipo="lluvia_detectada", severidad="critica", mensaje="m",
                                    fecha=datetime.utcnow(), atendida=False),
    )
    texto = generar_reporte_lote(datos)
    assert "3 alertas" in texto
    assert "exceso de temperatura" in texto.lower()
    assert "lluvia detectada" in texto.lower()
    assert "sigue sin atenderse" in texto
    # severidad crítica presente -> el cierre debe reflejar urgencia
    assert "cuanto antes" in texto


def test_reporte_lote_finalizado_usa_duracion_real_no_reloj_actual():
    inicio = datetime(2026, 1, 1, 0, 0, 0)
    fin = datetime(2026, 1, 9, 4, 0, 0)  # 8 dias y 4 horas = 196 horas
    datos = _datos_base(fecha_inicio_secado=inicio, fecha_fin_secado=fin, estado="finalizado")
    texto = generar_reporte_lote(datos)
    assert "terminó su secado" in texto
    assert "196 horas" in texto


def test_reporte_riesgo_lluvia_true_y_false():
    pred_riesgo = UltimaPrediccion(tiempo_estimado_horas=None, calidad_estimada=None, confianza=None,
                                    riesgo_lluvia_proxima=True, horas_anticipacion_lluvia=3, fecha=None)
    texto_riesgo = generar_reporte_lote(_datos_base(ultima_prediccion=pred_riesgo))
    assert "riesgo de lluvia" in texto_riesgo.lower()
    assert "cubrir el lote" in texto_riesgo

    pred_sin_riesgo = UltimaPrediccion(tiempo_estimado_horas=None, calidad_estimada=None, confianza=None,
                                        riesgo_lluvia_proxima=False, horas_anticipacion_lluvia=3, fecha=None)
    texto_sin_riesgo = generar_reporte_lote(_datos_base(ultima_prediccion=pred_sin_riesgo))
    assert "no estima riesgo de lluvia" in texto_sin_riesgo.lower()


def test_reporte_con_prediccion_completa_menciona_tiempo_y_calidad():
    pred = UltimaPrediccion(tiempo_estimado_horas=120.5, calidad_estimada="buena", confianza=82.3,
                             riesgo_lluvia_proxima=None, horas_anticipacion_lluvia=None, fecha=None)
    texto = generar_reporte_lote(_datos_base(ultima_prediccion=pred))
    assert "tiempo restante estimado" in texto
    assert "calidad final estimada de 'buena'" in texto
    assert "82%" in texto


# --- Integración: endpoint + historial, con BD -------------------------------------------------

def test_endpoint_reporte_respeta_dueno():
    _seed_lote_nlp()

    r_ok = client.get(f"/api/v1/anomalies/{ID_LOTE_NLP}/reporte", params={"id_usuario": USUARIO_NLP})
    assert r_ok.status_code == 200
    data = r_ok.json()
    assert data["id_lote"] == ID_LOTE_NLP
    assert isinstance(data["reporte_texto"], str) and len(data["reporte_texto"]) > 0
    assert data["id_reporte"] > 0

    r_ajeno = client.get(f"/api/v1/anomalies/{ID_LOTE_NLP}/reporte", params={"id_usuario": USUARIO_OTRO})
    assert r_ajeno.status_code == 403

    r_inexistente = client.get("/api/v1/anomalies/999999/reporte", params={"id_usuario": USUARIO_NLP})
    assert r_inexistente.status_code == 404


def test_historial_de_reportes_se_acumula_y_refleja_cambios():
    _seed_lote_nlp()

    r1 = client.get(f"/api/v1/anomalies/{ID_LOTE_NLP}/reporte", params={"id_usuario": USUARIO_NLP})
    texto_1 = r1.json()["reporte_texto"]

    db = SessionLocal()
    try:
        db.add(Alerta(id_lote=ID_LOTE_NLP, id_sensor=9001, tipo_alerta="temperatura_alta",
                       mensaje="m", nivel_severidad="alta"))
        db.commit()
    finally:
        db.close()

    r2 = client.get(f"/api/v1/anomalies/{ID_LOTE_NLP}/reporte", params={"id_usuario": USUARIO_NLP})
    texto_2 = r2.json()["reporte_texto"]
    assert texto_2 != texto_1  # la alerta nueva debe reflejarse en el segundo reporte
    assert "exceso de temperatura" in texto_2.lower()

    r_hist = client.get(f"/api/v1/anomalies/{ID_LOTE_NLP}/reportes", params={"id_usuario": USUARIO_NLP})
    assert r_hist.status_code == 200
    historial = r_hist.json()
    assert len(historial) >= 2
    ids = [h["id_reporte"] for h in historial]
    assert ids == sorted(ids, reverse=True)  # más reciente primero

    r_hist_ajeno = client.get(f"/api/v1/anomalies/{ID_LOTE_NLP}/reportes", params={"id_usuario": USUARIO_OTRO})
    assert r_hist_ajeno.status_code == 403


# --- Paso 4 (opción A): buscador de reportes históricos con BM25 -------------------------------
#
# Dos niveles, igual que el resto del archivo: unitarias sobre NLP/buscar_reportes.py (sin BD,
# corpus sintéticos) que cubren los casos borde de BM25 encontrados al construirlo -- corpus
# vacío/query vacío, corpus de 1 documento (IDF degenerado), empate exacto de frecuencia (IDF=0
# en TODOS los documentos), BM25 normal con corpus suficientemente grande, sin relación real, y
# el límite de `top_n`; y de integración sobre el endpoint (con BD), que confirma que sí
# encuentra lo relevante Y que nunca cruza reportes entre usuarios.

def test_buscar_reportes_corpus_o_query_vacio_regresa_vacio():
    assert buscar_reportes([], "lluvia") == []
    assert buscar_reportes([(1, "algo de lluvia")], "") == []
    assert buscar_reportes([(1, "algo de lluvia")], "   ") == []


def test_buscar_reportes_un_solo_documento_usa_fallback_y_encuentra():
    # Con 1 solo documento, BM25 puro da IDF negativo para toda palabra (ver docstring de
    # buscar_reportes) -- confirma que el fallback de coincidencia de tokens rescata este caso,
    # el más común para un usuario que apenas empieza.
    corpus = [(1, "Lluvia detectada crítica y exceso de temperatura también.")]
    resultados = buscar_reportes(corpus, "lluvia")
    assert len(resultados) == 1
    assert resultados[0].id_reporte == 1


def test_buscar_reportes_empate_exacto_de_bm25_cae_a_fallback():
    # 4 documentos, "lluvia detectada" aparece en exactamente 2 de 4: el IDF de BM25 da
    # matemáticamente 0 en ambos términos, así que el score puro de BM25 sería 0 en los 4 --
    # confirma que el fallback de coincidencia de tokens rescata el caso en vez de regresar
    # vacío (bug real encontrado al probar el endpoint con datos de ejemplo).
    corpus = [
        (1, "El lote 1 lleva 20 horas en secado. Se registraron 2 alertas: 1 de lluvia detectada, 1 de exceso de temperatura."),
        (2, "El lote 1 lleva 30 horas en secado. Se registraron 3 alertas: 2 de lluvia detectada, 1 de sensor con error."),
        (3, "El lote 2 lleva 5 horas en secado. No se han registrado alertas durante este proceso."),
        (4, "El lote 2 terminó su secado. Se registraron 2 alertas: 2 de secado estancado."),
    ]
    resultados = buscar_reportes(corpus, "lluvia detectada")
    assert {r.id_reporte for r in resultados} == {1, 2}


def test_buscar_reportes_bm25_normal_rankea_por_relevancia():
    corpus = [
        (1, "El lote A lleva 20 horas en secado. Se registraron 2 alertas: 1 de lluvia detectada, 1 de exceso de temperatura."),
        (2, "El lote B lleva 5 horas en secado. No se han registrado alertas durante este proceso."),
        (3, "El lote C terminó su secado. Se registraron 3 alertas: 3 de secado estancado."),
        (4, "El lote D lleva 40 horas en secado. Se registraron 1 alerta: 1 de lluvia detectada."),
        (5, "El lote E lleva 10 horas en secado. No se han registrado alertas durante este proceso."),
        (6, "El lote F terminó su secado. Se registraron 2 alertas: 2 de exceso de temperatura."),
    ]
    resultados = buscar_reportes(corpus, "lluvia")
    assert [r.id_reporte for r in resultados] == [4, 1]  # los 2 que mencionan lluvia, ordenados
    assert all(r.score > 0 for r in resultados)


def test_buscar_reportes_sin_relacion_real_regresa_vacio():
    corpus = [(1, "todo normal"), (2, "todo bien"), (3, "sin novedad")]
    assert buscar_reportes(corpus, "terremoto volcán") == []


def test_buscar_reportes_top_n_limita_resultados():
    corpus = [(i, f"reporte con lluvia número {i}") for i in range(1, 8)]
    resultados = buscar_reportes(corpus, "lluvia", top_n=2)
    assert len(resultados) == 2


USUARIO_BUSCADOR_A = 9101
USUARIO_BUSCADOR_B = 9102
ID_LOTE_BUSCADOR_A1 = 9101
ID_LOTE_BUSCADOR_A2 = 9102
ID_LOTE_BUSCADOR_B1 = 9103


def _seed_lotes_buscador():
    db = SessionLocal()
    try:
        if db.query(LoteCafe).filter(LoteCafe.id_lote == ID_LOTE_BUSCADOR_A1).first():
            return
        db.add(Sensor(id_sensor=ID_LOTE_BUSCADOR_A1, mac_address="AA:BB:CC:09:10:01", tipo="ambos", modelo="BMP280+DS18B20"))
        db.add(Sensor(id_sensor=ID_LOTE_BUSCADOR_A2, mac_address="AA:BB:CC:09:10:02", tipo="ambos", modelo="BMP280+DS18B20"))
        db.add(Sensor(id_sensor=ID_LOTE_BUSCADOR_B1, mac_address="AA:BB:CC:09:10:03", tipo="ambos", modelo="BMP280+DS18B20"))
        db.add(LoteCafe(id_lote=ID_LOTE_BUSCADOR_A1, id_usuario=USUARIO_BUSCADOR_A, id_sensor=ID_LOTE_BUSCADOR_A1,
                         nombre_lote="Lote A1", codigo_qr="QR-BUS-A1", tipo_proceso="lavado"))
        db.add(LoteCafe(id_lote=ID_LOTE_BUSCADOR_A2, id_usuario=USUARIO_BUSCADOR_A, id_sensor=ID_LOTE_BUSCADOR_A2,
                         nombre_lote="Lote A2", codigo_qr="QR-BUS-A2", tipo_proceso="lavado"))
        db.add(LoteCafe(id_lote=ID_LOTE_BUSCADOR_B1, id_usuario=USUARIO_BUSCADOR_B, id_sensor=ID_LOTE_BUSCADOR_B1,
                         nombre_lote="Lote B1", codigo_qr="QR-BUS-B1", tipo_proceso="lavado"))
        db.commit()

        # A propósito replica el empate exacto de 50% dentro de los reportes del usuario A (2 de
        # 4 reportes mencionan "lluvia detectada"), para probar el fallback también end-to-end
        # a través del endpoint, no solo de forma unitaria.
        db.add(ReporteLote(id_lote=ID_LOTE_BUSCADOR_A1, reporte_texto=(
            "El lote A1 lleva 20 horas en secado. Se registraron 2 alertas: "
            "1 de lluvia detectada, 1 de exceso de temperatura."
        )))
        db.add(ReporteLote(id_lote=ID_LOTE_BUSCADOR_A1, reporte_texto=(
            "El lote A1 lleva 30 horas en secado. Se registraron 3 alertas: "
            "2 de lluvia detectada, 1 de sensor con error."
        )))
        db.add(ReporteLote(id_lote=ID_LOTE_BUSCADOR_A2, reporte_texto=(
            "El lote A2 lleva 5 horas en secado. No se han registrado alertas durante este proceso."
        )))
        db.add(ReporteLote(id_lote=ID_LOTE_BUSCADOR_A2, reporte_texto=(
            "El lote A2 terminó su secado. Se registraron 2 alertas: 2 de secado estancado."
        )))
        db.add(ReporteLote(id_lote=ID_LOTE_BUSCADOR_B1, reporte_texto=(
            "El lote de otro usuario tuvo lluvia detectada crítica y exceso de temperatura también."
        )))
        db.commit()
    finally:
        db.close()


def test_endpoint_buscar_reportes_encuentra_lo_relevante_y_aisla_por_usuario():
    _seed_lotes_buscador()

    r = client.get("/api/v1/anomalies/reportes/buscar", params={"id_usuario": USUARIO_BUSCADOR_A, "query": "lluvia detectada"})
    assert r.status_code == 200
    resultados = r.json()
    assert len(resultados) == 2
    assert {item["id_lote"] for item in resultados} == {ID_LOTE_BUSCADOR_A1}
    for item in resultados:
        assert "lluvia" in item["reporte_texto"].lower()

    # Usuario B: un solo reporte propio que sí menciona lluvia (corpus chico -> fallback).
    r_b = client.get("/api/v1/anomalies/reportes/buscar", params={"id_usuario": USUARIO_BUSCADOR_B, "query": "lluvia"})
    assert r_b.status_code == 200
    resultados_b = r_b.json()
    assert len(resultados_b) == 1
    assert resultados_b[0]["id_lote"] == ID_LOTE_BUSCADOR_B1

    # Aislamiento: los ids de reporte que ve B nunca deben ser los que ve A, aunque el término
    # buscado ("lluvia") aparezca en reportes de ambos usuarios.
    ids_reporte_a = {item["id_reporte"] for item in resultados}
    ids_reporte_b = {item["id_reporte"] for item in resultados_b}
    assert ids_reporte_a.isdisjoint(ids_reporte_b)


def test_endpoint_buscar_reportes_sin_coincidencia_o_sin_historial_regresa_vacio():
    _seed_lotes_buscador()

    r_sin_match = client.get(
        "/api/v1/anomalies/reportes/buscar", params={"id_usuario": USUARIO_BUSCADOR_A, "query": "terremoto volcán"}
    )
    assert r_sin_match.status_code == 200
    assert r_sin_match.json() == []

    r_sin_reportes = client.get(
        "/api/v1/anomalies/reportes/buscar", params={"id_usuario": 999999, "query": "lluvia"}
    )
    assert r_sin_reportes.status_code == 200
    assert r_sin_reportes.json() == []


# --- Opción B: clasificador de texto ligero (TF-IDF + Naive Bayes) ------------------------------
#
# NLP/preparar_datos_clasificador.py lee TODA la tabla `alertas` sin filtrar por lote/usuario (a
# propósito: el clasificador es global, no por usuario, a diferencia del resto del PLN de este
# archivo). Usar la BD compartida del resto de la suite mezclaría alertas sembradas por
# test_api.py y por las pruebas de arriba con las de aquí, haciendo no deterministas las
# aserciones de conteo/clases -- por eso estas pruebas usan una BD sqlite PROPIA, aislada, en vez
# de `SessionLocal`.

def _sesion_aislada():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine_aislado = create_engine(f"sqlite:///{tmp.name}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine_aislado)
    return sessionmaker(bind=engine_aislado)()


def _sembrar_alertas(db, n, con_mensaje_nulo=0):
    plantillas = [
        ("lluvia_detectada", "Riesgo crítico: se requiere atención inmediata. Lluvia detectada sobre el lote en secado.", "critica"),
        ("temperatura_alta", "Patrón de riesgo: se recomienda atender el lote pronto. Temperatura del grano por encima del rango ideal.", "alta"),
    ]
    for i in range(n):
        tipo, mensaje, severidad = plantillas[i % len(plantillas)]
        db.add(Alerta(id_lote=1, id_sensor=1, tipo_alerta=tipo, mensaje=mensaje, nivel_severidad=severidad))
    for _ in range(con_mensaje_nulo):
        db.add(Alerta(id_lote=1, id_sensor=1, tipo_alerta="temperatura_alta", mensaje=None, nivel_severidad="alta"))
    db.commit()


def test_recolectar_ejemplos_ignora_mensajes_nulos_y_respeta_umbral():
    db = _sesion_aislada()
    try:
        assert recolectar_ejemplos(db) == ([], [])
        assert hay_suficientes_datos(db) is False

        _sembrar_alertas(db, n=MIN_EJEMPLOS_ENTRENAMIENTO - 1, con_mensaje_nulo=3)
        textos, etiquetas = recolectar_ejemplos(db)
        assert len(textos) == MIN_EJEMPLOS_ENTRENAMIENTO - 1  # los de mensaje=None no cuentan
        assert len(etiquetas) == len(textos)
        assert hay_suficientes_datos(db) is False  # todavía por debajo del umbral

        _sembrar_alertas(db, n=1)  # llega justo al umbral
        assert hay_suficientes_datos(db) is True
    finally:
        db.close()


def test_entrenar_sin_suficientes_datos_no_crea_artefacto(tmp_path):
    db = _sesion_aislada()
    try:
        _sembrar_alertas(db, n=MIN_EJEMPLOS_ENTRENAMIENTO - 1)
        ruta_modelo = str(tmp_path / "clasificador.joblib")
        ruta_metricas = str(tmp_path / "metricas.json")

        resultado = entrenar(db, ruta_modelo=ruta_modelo, ruta_metricas=ruta_metricas)
        assert resultado is None
        assert not (tmp_path / "clasificador.joblib").exists()
        assert not (tmp_path / "metricas.json").exists()
    finally:
        db.close()


def test_entrenar_con_suficientes_datos_guarda_artefacto_y_metricas(tmp_path):
    db = _sesion_aislada()
    try:
        _sembrar_alertas(db, n=MIN_EJEMPLOS_ENTRENAMIENTO + 5)
        ruta_modelo = str(tmp_path / "clasificador.joblib")
        ruta_metricas = str(tmp_path / "metricas.json")

        resultado = entrenar(db, ruta_modelo=ruta_modelo, ruta_metricas=ruta_metricas)
        assert resultado is not None
        assert resultado["n_ejemplos"] == MIN_EJEMPLOS_ENTRENAMIENTO + 5
        assert set(resultado["clases"]) == {"alta", "critica"}
        # Mensajes de plantilla fija -> memorización esperada, ver el aviso documentado.
        assert resultado["exactitud_train"] == 1.0
        assert (tmp_path / "clasificador.joblib").exists()
        assert (tmp_path / "metricas.json").exists()
    finally:
        db.close()


def test_clasificador_texto_sin_artefacto_regresa_none(tmp_path):
    clasificador = ClasificadorTexto(ruta_modelo=str(tmp_path / "no_existe.joblib"))
    assert clasificador.disponible() is False
    assert clasificador.clasificar("hay lluvia sobre el lote") is None
    assert clasificador.clasificar("") is None
    assert clasificador.clasificar("   ") is None


def test_clasificador_texto_con_artefacto_entrenado_clasifica_texto_libre(tmp_path):
    db = _sesion_aislada()
    try:
        _sembrar_alertas(db, n=MIN_EJEMPLOS_ENTRENAMIENTO + 2)
        ruta_modelo = str(tmp_path / "clasificador.joblib")
        entrenar(db, ruta_modelo=ruta_modelo, ruta_metricas=str(tmp_path / "metricas.json"))
    finally:
        db.close()

    clasificador = ClasificadorTexto(ruta_modelo=ruta_modelo)
    assert clasificador.disponible() is True

    resultado = clasificador.clasificar("hay agua encima del café y está lloviendo mucho")
    assert resultado is not None
    assert resultado["severidad_sugerida"] in {"alta", "critica"}
    assert 0.0 <= resultado["confianza"] <= 100.0
    assert set(resultado["probabilidades"].keys()) == {"alta", "critica"}
    assert abs(sum(resultado["probabilidades"].values()) - 100.0) < 0.1


def test_endpoint_clasificar_texto_sin_modelo_responde_no_disponible(monkeypatch, tmp_path):
    from app.api.routes import nlp as nlp_route

    monkeypatch.setattr(nlp_route, "clasificador", ClasificadorTexto(ruta_modelo=str(tmp_path / "no_existe.joblib")))

    r = client.post("/api/v1/nlp/clasificar-texto", json={"id_usuario": 1, "texto": "hay lluvia"})
    assert r.status_code == 200
    body = r.json()
    assert body["disponible"] is False
    assert "mensaje" in body


def test_endpoint_clasificar_texto_con_modelo_responde_prediccion_y_valida_texto_vacio(monkeypatch, tmp_path):
    db = _sesion_aislada()
    try:
        _sembrar_alertas(db, n=MIN_EJEMPLOS_ENTRENAMIENTO + 2)
        ruta_modelo = str(tmp_path / "clasificador.joblib")
        resultado_entrenamiento = entrenar(db, ruta_modelo=ruta_modelo, ruta_metricas=str(tmp_path / "metricas.json"))
        assert resultado_entrenamiento is not None
    finally:
        db.close()

    from app.api.routes import nlp as nlp_route
    monkeypatch.setattr(nlp_route, "clasificador", ClasificadorTexto(ruta_modelo=ruta_modelo))

    r = client.post(
        "/api/v1/nlp/clasificar-texto",
        json={"id_usuario": 1, "texto": "hay agua encima del café y está lloviendo mucho"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["disponible"] is True
    assert body["severidad_sugerida"] in {"alta", "critica"}
    assert 0.0 <= body["confianza"] <= 100.0

    r_vacio = client.post("/api/v1/nlp/clasificar-texto", json={"id_usuario": 1, "texto": ""})
    assert r_vacio.status_code == 422  # min_length=1 en el schema, ni llega al clasificador
