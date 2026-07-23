#!/usr/bin/env python3
# Archivo: scripts/e2e_nlp_produccion.py
"""
Prueba end-to-end de NLP/NLG (reportes, buscador BM25, clasificador de texto) Y del pipeline de
ML/AG que los alimenta, contra la base de datos REAL de producción (Neon) -- no contra el sqlite
de `pytest`. Pensado para correrse MANUALMENTE, una vez, desde una máquina con acceso de red a
Neon (el entorno donde se escribió este script -- el sandbox de Cowork -- no tiene salida de red
hacia el host de Neon, por eso este script queda aquí para que tú lo corras).

Cómo correrlo:
    export DATABASE_URL="postgresql://...tu cadena real de Neon..."
    python3 scripts/e2e_nlp_produccion.py

Qué hace, en orden:
  1. Verifica que existan las tablas necesarias (si `migration.sql` no se ha aplicado todavía en
     Neon, ABORTA con un mensaje claro en vez de fallar a medias o crear tablas por su cuenta --
     a propósito NO llama a `init_db()`/`create_all` contra producción: crear el esquema real es
     responsabilidad de correr `migration.sql` a mano, revisado, no de que este script infiera
     una versión "parecida" vía el ORM).
  2. Crea un lote y usuario de prueba CLARAMENTE marcados (`nombre_lote` empieza con
     "TEST-E2E-BORRAR", IDs en un rango alto reservado que no debería chocar nunca con datos
     reales) -- nunca toca lotes/usuarios existentes. Si por cualquier razón ya existe algo con
     ese id_lote, aborta sin tocar nada en vez de sobrescribir.
  3. Manda 7 lecturas variadas al pipeline completo real (reglas de dominio + IsolationForest +
     RandomForest + Algoritmo Genético de lluvia), igual que lo haría un ESP32 real vía el Gestor.
  4. Genera un reporte NLG sobre ese lote y confirma que el resumen extractivo BM25 (Nivel 2) se
     activa (>5 alertas generadas por las 7 lecturas de arriba).
  5. Prueba el buscador de historial (BM25, opción A) buscando sobre ese mismo reporte.
  6. Prueba el clasificador de texto ligero (opción B) SOLO SI ya existe un artefacto entrenado
     en `NLP/artifacts/` -- este script NO entrena nada (entrenar es una acción aparte y
     deliberada, ver `python3 -m NLP.entrenar_clasificador_texto`); si no existe, lo reporta como
     estado esperado, no como falla.
  7. Limpia TODO lo que este script creó (alertas, predicciones, recomendaciones, reportes,
     lecturas, lote, sensor) SIEMPRE, incluso si algún paso falló a medias (bloque `finally`),
     filtrando todo por el id_lote de prueba -- no debe quedar ni un rastro en producción.

Nota sobre el poller en paralelo: si el servicio real está corriendo en producción con
`POLLING_ENABLED=true` mientras corres este script, es posible (aunque de bajo riesgo) que el
poller también procese alguna de las lecturas de prueba por su cuenta, generando una predicción o
alerta extra para el MISMO id_lote de prueba -- la limpieza del paso 7 la atrapa igual, porque
filtra por id_lote, no por "lo que este script recuerda haber creado". No se manda ningún push
real: el usuario de prueba nunca tiene un dispositivo registrado en `dispositivos_usuario`.
"""
import os
import sys
import time

# Permite correr este script como `python3 scripts/e2e_nlp_produccion.py` desde la raíz del
# proyecto (o desde cualquier otro directorio) sin depender de PYTHONPATH ya configurado --
# mismo criterio que scripts/recolectar_datos_reales.py.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import inspect  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.api.routes.inference import ejecutar_pipeline
from app.models.alertas import Alerta
from app.models.database import SessionLocal, engine
from app.models.lecturas_ambientales import LecturaAmbiental
from app.models.lotes_cafe import LoteCafe
from app.models.predicciones import Prediccion
from app.models.recomendaciones import Recomendacion
from app.models.reportes_lote import ReporteLote
from app.models.sensores import Sensor
from app.services.preprocessor import Preprocessor
from NLP.buscar_reportes import buscar_reportes
from NLP.clasificar_texto import ClasificadorTexto
from NLP.generar_reporte import generar_reporte_lote
from NLP.recopilar_datos_reporte import recopilar_datos_lote
from NLP.registrar_reporte import guardar_reporte

TABLAS_REQUERIDAS = [
    "lotes_cafe", "sensores", "alertas", "predicciones", "recomendaciones",
    "reportes_lote", "dispositivos_usuario", "ml_estado_polling", "retroalimentacion_ml",
    "lecturas_ambientales",
]

# Rango reservado, muy por encima de cualquier ID real esperado en este proyecto -- si algún día
# coincidiera con datos reales, `confirmar_no_colision` aborta antes de tocar nada.
ID_USUARIO_TEST = 999_990_001
ID_LOTE_TEST = 999_990_001
ID_SENSOR_TEST = 999_990_001


def verificar_esquema() -> bool:
    insp = inspect(engine)
    existentes = set(insp.get_table_names())
    faltantes = [t for t in TABLAS_REQUERIDAS if t not in existentes]
    if faltantes:
        print(f"[ABORTA] Faltan tablas en la BD: {faltantes}")
        print("Corre migration.sql contra Neon antes de este script (ver README.md, sección 6 'Pendientes conocidos').")
        return False
    print("[OK] Todas las tablas requeridas existen en la BD.")
    return True


def confirmar_no_colision(db: Session) -> bool:
    if db.query(LoteCafe).filter(LoteCafe.id_lote == ID_LOTE_TEST).first() is not None:
        print(f"[ABORTA] Ya existe un lote con id_lote={ID_LOTE_TEST} -- no se toca ni se "
              "sobreescribe. Revisa manualmente qué es antes de reintentar.")
        return False
    return True


def crear_lote_prueba(db: Session) -> None:
    db.add(Sensor(id_sensor=ID_SENSOR_TEST, mac_address="TEST-E2E-BORRAR", tipo="ambos", modelo="TEST"))
    db.add(LoteCafe(
        id_lote=ID_LOTE_TEST, id_usuario=ID_USUARIO_TEST, id_sensor=ID_SENSOR_TEST,
        nombre_lote="TEST-E2E-BORRAR", codigo_qr=f"QR-TEST-E2E-{int(time.time())}",
        tipo_proceso="lavado",
    ))
    db.commit()
    print(f"[OK] Lote de prueba creado: id_lote={ID_LOTE_TEST}, id_usuario={ID_USUARIO_TEST}")


def correr_pipeline_ml(db: Session) -> None:
    pre = Preprocessor()
    casos = [
        ("lectura normal", dict(temperatura_grano=27.0, temperatura_ambiental=25.0, humedad_grano=2000.0, lluvia=0.0, luz=40000.0)),
        ("temperatura alta critico", dict(temperatura_grano=42.0, temperatura_ambiental=25.0, humedad_grano=2000.0, lluvia=0.0, luz=40000.0)),
        ("lluvia detectada", dict(temperatura_grano=27.0, temperatura_ambiental=25.0, humedad_grano=2000.0, lluvia=1.0, luz=40000.0)),
        ("temperatura alta riesgo", dict(temperatura_grano=39.0, temperatura_ambiental=25.0, humedad_grano=2000.0, lluvia=0.0, luz=40000.0)),
        ("valor imposible caliente", dict(temperatura_grano=90.0, temperatura_ambiental=25.0, humedad_grano=2000.0, lluvia=0.0, luz=40000.0)),
        ("valor imposible frio", dict(temperatura_grano=-20.0, temperatura_ambiental=25.0, humedad_grano=2000.0, lluvia=0.0, luz=40000.0)),
        ("lluvia otra vez", dict(temperatura_grano=27.0, temperatura_ambiental=25.0, humedad_grano=2000.0, lluvia=1.0, luz=40000.0)),
    ]
    lote = db.query(LoteCafe).filter(LoteCafe.id_lote == ID_LOTE_TEST).first()
    for etiqueta, lecturas in casos:
        features = pre.transform(lecturas)
        resultado = ejecutar_pipeline(
            db, lote, ID_LOTE_TEST, "lavado", ID_SENSOR_TEST, features, horas_transcurridas=10.0,
            guardar_lectura=True,
        )
        print(f"  [{etiqueta}] severidad={resultado.nivel_severidad} "
              f"es_anomalia={resultado.es_anomalia} alerta_generada={resultado.alerta_generada}")
    print("[OK] Pipeline de ML/AG corrido contra Neon real (7 lecturas).")


def probar_reporte_nlg(db: Session) -> None:
    datos = recopilar_datos_lote(db, ID_LOTE_TEST)
    assert datos is not None, "recopilar_datos_lote regresó None para un lote que sí existe -- revisar"
    texto = generar_reporte_lote(datos)
    print("\n[REPORTE NLG GENERADO CONTRA NEON REAL]")
    print(texto)
    assert "Eventos más relevantes según el modelo" in texto, (
        "No se activó el resumen extractivo BM25 (Nivel 2) -- se esperaban >5 alertas."
    )
    print("\n[OK] NLG genera el reporte y BM25 Nivel 2 (resumen extractivo) se activó correctamente.")
    id_reporte = guardar_reporte(db, ID_LOTE_TEST, texto)
    print(f"[OK] Reporte guardado en reportes_lote (id_reporte={id_reporte}).")


def probar_buscador(db: Session) -> None:
    filas = (
        db.query(ReporteLote)
        .join(LoteCafe, ReporteLote.id_lote == LoteCafe.id_lote)
        .filter(LoteCafe.id_usuario == ID_USUARIO_TEST)
        .all()
    )
    corpus = [(f.id_reporte, f.reporte_texto) for f in filas]
    resultados = buscar_reportes(corpus, "lluvia critico")
    print(f"\n[BUSCADOR BM25] {len(resultados)} resultado(s) para 'lluvia critico'")
    for r in resultados:
        print(f"  id_reporte={r.id_reporte} score={r.score:.3f}")
    assert len(resultados) >= 1, "El buscador no encontró el reporte recién creado -- revisar."
    print("[OK] Buscador de historial (opción A) funciona contra Neon real.")


def probar_clasificador() -> None:
    clasificador = ClasificadorTexto()
    if not clasificador.disponible():
        print(
            "\n[CLASIFICADOR] No hay artefacto entrenado todavía en producción "
            "(NLP/artifacts/clasificador_texto.joblib no existe). Esperado si aún no hay >= 10 "
            "alertas reales o no se ha corrido `python3 -m NLP.entrenar_clasificador_texto` en "
            "este servidor -- NO es una falla de este E2E."
        )
        return
    resultado = clasificador.clasificar("hay agua encima del café y está lloviendo mucho")
    print(f"\n[CLASIFICADOR] {resultado}")
    print("[OK] Clasificador de texto (opción B) responde contra el artefacto real de producción.")


def limpiar(db: Session) -> None:
    db.query(Recomendacion).filter(Recomendacion.id_lote == ID_LOTE_TEST).delete()
    db.query(Prediccion).filter(Prediccion.id_lote == ID_LOTE_TEST).delete()
    db.query(Alerta).filter(Alerta.id_lote == ID_LOTE_TEST).delete()
    db.query(ReporteLote).filter(ReporteLote.id_lote == ID_LOTE_TEST).delete()
    db.query(LecturaAmbiental).filter(LecturaAmbiental.id_lote == ID_LOTE_TEST).delete()
    db.query(LoteCafe).filter(LoteCafe.id_lote == ID_LOTE_TEST).delete()
    db.query(Sensor).filter(Sensor.id_sensor == ID_SENSOR_TEST).delete()
    db.commit()
    print("\n[OK] Limpieza completa -- no queda ningún dato de prueba en la BD de producción.")


def main() -> int:
    backend = engine.url.get_backend_name()
    if backend == "sqlite":
        print(
            "[AVISO] DATABASE_URL apunta a sqlite, no a Neon/Postgres. Este script está pensado "
            "para validar la BD real de producción -- exporta la DATABASE_URL real antes de "
            "correrlo si esa es la intención."
        )
    else:
        print(f"[INFO] Conectando a: {engine.url.host} (base: {engine.url.database})")

    if not verificar_esquema():
        return 1

    db: Session = SessionLocal()
    creado: bool = False
    try:
        if not confirmar_no_colision(db):
            return 1
        crear_lote_prueba(db)
        creado = True
        correr_pipeline_ml(db)
        probar_reporte_nlg(db)
        probar_buscador(db)
        probar_clasificador()
        print("\n=== TODAS LAS PRUEBAS E2E PASARON CONTRA LA BASE DE DATOS DE PRODUCCIÓN ===")
        return 0
    except AssertionError as exc:
        print(f"\n[FALLO] {exc}")
        return 1
    except Exception as exc:
        print(f"\n[ERROR INESPERADO] {exc}")
        return 1
    finally:
        if creado:
            print("\nLimpiando datos de prueba...")
            limpiar(db)
        db.close()


if __name__ == "__main__":
    sys.exit(main())
