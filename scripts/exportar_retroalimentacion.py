#script/exportar_retroalimentacion.py
"""
Exporta la tabla retroalimentacion_ml (resultados reales reportados por productores vía
POST /internal/lotes/{id_lote}/resultado-real, RNF-19) al mismo esquema de columnas que
usa scripts/train_models.py, para poder combinarla con el dataset sintético al reentrenar.

Cada fila de retroalimentacion_ml es un lote ya finalizado: se le asigna horas_transcurridas
= tiempo_real_horas y horas_restantes = 0, y se corren las MISMAS reglas de dominio
(app/services/rules.py) sobre sus features para obtener _es_anomalia/_severidad/_tipo_anomalia,
igual que hace generar_dataset.py con los datos sintéticos.

Salida: data/raw/retroalimentacion_real.csv (solo encabezados si aún no hay reportes reales).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.models.database import SessionLocal, init_db  # noqa: E402
from app.models.retroalimentacion_ml import RetroalimentacionML  # noqa: E402
from app.services.rules import evaluar_lectura  # noqa: E402

COLUMNAS = [
    "id_lote", "tipo_proceso", "horas_transcurridas", "horas_restantes",
    "temperatura_grano", "temperatura_ambiental", "humedad_ambiental", "humedad_grano", "lluvia", "luz",
    "_es_anomalia", "_severidad", "_tipo_anomalia", "_calidad_final_lote",
]


def main():
    init_db()
    db = SessionLocal()
    try:
        filas_bd = db.query(RetroalimentacionML).all()
    finally:
        db.close()

    filas = []
    for r in filas_bd:
        features = {
            "temperatura_grano": float(r.temperatura_grano) if r.temperatura_grano is not None else 0.0,
            "temperatura_ambiental": float(r.temperatura_ambiental) if r.temperatura_ambiental is not None else 0.0,
            "humedad_ambiental": float(r.humedad_ambiental) if r.humedad_ambiental is not None else 0.0,
            "humedad_grano": float(r.humedad_grano) if r.humedad_grano is not None else 0.0,
            "lluvia": float(r.lluvia) if r.lluvia is not None else 0.0,
            "luz": float(r.luz) if r.luz is not None else 0.0,
        }
        evaluacion = evaluar_lectura(r.tipo_proceso, features)
        filas.append({
            "id_lote": r.id_lote,
            "tipo_proceso": r.tipo_proceso,
            "horas_transcurridas": float(r.tiempo_real_horas),
            "horas_restantes": 0.0,
            **features,
            "_es_anomalia": evaluacion["es_anomalia"],
            "_severidad": evaluacion["severidad"],
            "_tipo_anomalia": evaluacion["tipo_principal"],
            "_calidad_final_lote": r.calidad_real,
        })

    df = pd.DataFrame(filas, columns=COLUMNAS)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "retroalimentacion_real.csv")
    df.to_csv(out_path, index=False)
    print(f"Filas reales exportadas: {len(df)}")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
