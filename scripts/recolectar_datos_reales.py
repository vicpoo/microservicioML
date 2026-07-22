#scripts/recolectar_datos_reales.py
"""
Recolección de datos (paso 2 del pipeline de ML) — REEMPLAZA a scripts/generar_dataset.py.

Ya no se simula nada: este script se conecta a la BD real (Neon, vía DATABASE_URL en
.env) y construye el dataset de entrenamiento a partir de las tablas que el
microservicio ya escribe/lee en producción:

  - lecturas_ambientales + lotes_cafe  -> features + horas_transcurridas por lectura,
    más las etiquetas _es_anomalia/_severidad/_tipo_anomalia (calculadas con las MISMAS
    reglas de dominio que usa el servicio en vivo, app/services/rules.py, para que
    modelo y reglas de producción nunca queden desalineados).
  - lotes_cafe.fecha_fin_secado         -> horas_restantes, SOLO para lotes finalizados
    con fecha de fin registrada (para los demás lotes queda NaN: no se puede conocer
    el futuro de un lote que sigue en proceso).
  - retroalimentacion_ml                -> calidad_final real por lote (RNF-19), SOLO
    para lotes que el productor ya reportó. Sin esto, calidad_final queda NaN.

IMPORTANTE (ver definicion_problema_kajve.md, Sección 6): al momento de escribir esto
hay un solo lote real con sensor físico, sin ningún ciclo completo todavía. Este script
va a correr y producir un CSV, pero con muy pocas filas y probablemente CERO lotes
usables para entrenar horas_restantes o calidad — eso es exactamente lo esperado en
esta etapa del piloto, no un bug. El valor de este script es que la plomería ya queda
lista: según se acumulen más lotes reales, train_models.py tendrá cada vez más señal
real y cada vez menos necesidad de heurísticas.

Salida: data/raw/lecturas_reales_entrenamiento.csv
"""
import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models.database import SessionLocal, init_db  # noqa: E402
from app.models.lecturas_ambientales import LecturaAmbiental  # noqa: E402
from app.models.lotes_cafe import LoteCafe  # noqa: E402
from app.models.retroalimentacion_ml import RetroalimentacionML  # noqa: E402
from app.services import rules  # noqa: E402

COLUMNAS_SALIDA = [
    "id_lote", "tipo_proceso", "horas_transcurridas", "horas_restantes",
    "temperatura_grano", "temperatura_ambiental", "humedad_grano", "lluvia", "luz", "delta_temp",
    "_es_anomalia", "_severidad", "_tipo_anomalia", "_calidad_final_lote",
]

MIN_LOTES_FINALIZADOS_SANO = 5  # umbral solo informativo, para la advertencia final


def _to_float(v):
    return float(v) if v is not None else None


def construir_dataset() -> pd.DataFrame:
    init_db()
    db = SessionLocal()
    try:
        lotes = {l.id_lote: l for l in db.query(LoteCafe).all()}
        lecturas = (
            db.query(LecturaAmbiental)
            .order_by(LecturaAmbiental.id_lote, LecturaAmbiental.timestamp)
            .all()
        )
        calidad_por_lote = {}
        for r in db.query(RetroalimentacionML).order_by(RetroalimentacionML.fecha_reporte.desc()).all():
            calidad_por_lote.setdefault(r.id_lote, r.calidad_real)  # se queda con la más reciente
    finally:
        db.close()

    # Agrupar lecturas por lote (ya vienen ordenadas por timestamp gracias al ORDER BY)
    lecturas_por_lote = {}
    for l in lecturas:
        lecturas_por_lote.setdefault(l.id_lote, []).append(l)

    filas = []
    for id_lote, lecturas_lote in lecturas_por_lote.items():
        lote = lotes.get(id_lote)
        if lote is None:
            continue  # lectura huérfana (no debería pasar, pero no se descarta el resto por esto)

        tipo_proceso = (lote.tipo_proceso or "lavado").lower()
        fecha_inicio = lote.fecha_inicio_secado
        fecha_fin = lote.fecha_fin_secado
        calidad_final = calidad_por_lote.get(id_lote)

        for i, lectura in enumerate(lecturas_lote):
            temp_grano = _to_float(lectura.temperatura_grano)
            temp_amb = _to_float(lectura.temperatura)
            humedad_grano_raw = _to_float(lectura.humedad_grano)
            luz = _to_float(lectura.luz)
            lluvia = 1.0 if lectura.lluvia_detectada else 0.0

            # delta_temp_reciente: contra la lectura inmediatamente anterior del mismo lote
            delta_temp_reciente = None
            if i > 0 and lecturas_lote[i - 1].temperatura_grano is not None and temp_grano is not None:
                delta_temp_reciente = abs(temp_grano - float(lecturas_lote[i - 1].temperatura_grano))

            # delta_humedad_grano_24h_pct: contra la lectura más cercana a 24h antes,
            # convertido a % (None si no hay calibración, igual que en producción).
            delta_humedad_grano_24h_pct = None
            limite_24h = lectura.timestamp - timedelta(hours=24)
            candidatas = [x for x in lecturas_lote[:i] if x.timestamp <= limite_24h]
            if candidatas:
                ref = candidatas[-1]  # la más cercana a 24h (última antes del límite)
                pct_antiguo = rules.humedad_grano_raw_a_porcentaje(_to_float(ref.humedad_grano))
                pct_actual = rules.humedad_grano_raw_a_porcentaje(humedad_grano_raw)
                if pct_antiguo is not None and pct_actual is not None:
                    delta_humedad_grano_24h_pct = pct_antiguo - pct_actual

            features = {
                "temperatura_grano": temp_grano or 0.0,
                "temperatura_ambiental": temp_amb or 0.0,
                "humedad_grano": humedad_grano_raw or 0.0,
                "lluvia": lluvia,
                "luz": luz or 0.0,
            }
            evaluacion = rules.evaluar_lectura(
                tipo_proceso, features,
                delta_temp_reciente=delta_temp_reciente,
                delta_humedad_grano_24h_pct=delta_humedad_grano_24h_pct,
            )

            horas_transcurridas = None
            if fecha_inicio is not None:
                horas_transcurridas = (lectura.timestamp - fecha_inicio).total_seconds() / 3600.0

            horas_restantes = None
            if fecha_fin is not None and lote.estado == "finalizado":
                horas_restantes = max((fecha_fin - lectura.timestamp).total_seconds() / 3600.0, 0.0)

            filas.append({
                "id_lote": id_lote,
                "tipo_proceso": tipo_proceso,
                "horas_transcurridas": horas_transcurridas,
                "horas_restantes": horas_restantes,
                "temperatura_grano": temp_grano,
                "temperatura_ambiental": temp_amb,
                "humedad_grano": humedad_grano_raw,
                "lluvia": lluvia,
                "luz": luz,
                "delta_temp": (temp_grano - temp_amb) if (temp_grano is not None and temp_amb is not None) else None,
                "_es_anomalia": evaluacion["es_anomalia"],
                "_severidad": evaluacion["severidad"],
                "_tipo_anomalia": evaluacion["tipo_principal"],
                "_calidad_final_lote": calidad_final,
            })

    return pd.DataFrame(filas, columns=COLUMNAS_SALIDA)


def main():
    df = construir_dataset()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lecturas_reales_entrenamiento.csv")
    df.to_csv(out_path, index=False)

    n_lotes = df["id_lote"].nunique() if not df.empty else 0
    n_lotes_con_fin = df.dropna(subset=["horas_restantes"])["id_lote"].nunique() if not df.empty else 0
    n_lotes_con_calidad = df.dropna(subset=["_calidad_final_lote"])["id_lote"].nunique() if not df.empty else 0

    print(f"Lecturas reales recolectadas: {len(df):,}")
    print(f"Lotes distintos: {n_lotes}")
    print(f"Lotes con horas_restantes conocidas (finalizados con fecha_fin_secado): {n_lotes_con_fin}")
    print(f"Lotes con calidad_final conocida (retroalimentacion_ml): {n_lotes_con_calidad}")
    print(f"Guardado en {out_path}")

    if n_lotes_con_fin < MIN_LOTES_FINALIZADOS_SANO or n_lotes_con_calidad < MIN_LOTES_FINALIZADOS_SANO:
        print(
            f"\nAVISO: menos de {MIN_LOTES_FINALIZADOS_SANO} lotes con etiqueta real de "
            "tiempo/calidad. Es normal en esta etapa del piloto (ver Sección 6 de "
            "definicion_problema_kajve.md); entrenar rf_tiempo_restante/rf_calidad con "
            "esto todavía no va a dar un modelo confiable. La detección de anomalías "
            "(isolation_forest / rf_tipo_anomalia) sí puede aprovechar todas las "
            "lecturas, tengan o no lote finalizado."
        )


if __name__ == "__main__":
    main()
