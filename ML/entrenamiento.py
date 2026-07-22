"""
ML/entrenamiento.py

Paso 8 del pipeline de ML — Entrenamiento: ajusta cada modelo elegido en el paso 7
(`07_seleccion_modelo.ipynb`) usando SOLO `data/processed/train.csv`. La evaluación real de qué
tan bien generalizan (con `test.csv`) es el paso 9, no este módulo -- aquí solo se reporta, como
mucho, un diagnóstico de ajuste sobre el propio train (útil para confirmar que el entrenamiento
corrió bien, NUNCA como evidencia de qué tan bueno es el modelo).

Modelos y su función de entrenamiento:
  - IsolationForest        -> entrenar_isolation_forest()
  - RandomForestClassifier (tipo de anomalía) -> entrenar_rf_tipo_anomalia()
  - Algoritmo Genético (lluvia próxima)       -> entrenar_ga_lluvia()
  - RandomForestRegressor (tiempo restante)   -> entrenar_regresor_tiempo() -- se omite hoy
    (0 lotes finalizados con horas_restantes conocida; guard MIN_LOTES_TIEMPO).
  - RandomForestClassifier (calidad final)    -> entrenar_clasificador_calidad() -- se omite hoy
    (la columna _calidad_final_lote ni siquiera existe todavía en este dataset offline; la tabla
    retroalimentacion_ml real sigue vacía en Neon).
  - Recomendaciones: no aplica, no hay nada que entrenar (mapeo determinístico por reglas).

Los artefactos se guardan en ML/artifacts/ (NO en app/ml/artifacts/, que es donde vive el
pipeline de producción con su propio esquema de columnas -- ver README.md de esta carpeta para
la nota sobre por qué todavía no están unificados).
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ML import prediccion_lluvia_ga as ga

# NOTA IMPORTANTE: el nombre de columna aquí es "humedad_grano" (no "humedad_grano_raw" como en
# los notebooks 02-06 de esta carpeta) A PROPÓSITO -- así el Pipeline/IsolationForest queda
# ajustado esperando el MISMO nombre de columna que construyen app/services/anomaly_detector.py
# y app/services/predictor.py en producción (_fila_ml/_fila usan "humedad_grano"). Si se
# entrenara con "humedad_grano_raw", el modelo cargaría bien pero fallaría en cada predicción
# real con "columns are missing: {'humedad_grano_raw'}" -- justo el bug que se encontró y
# corrigió aquí al conectar el paso 8/10 con el paso 11 (despliegue).
NUMERIC_FEATURES = ["temperatura_grano", "temperatura_ambiental", "humedad_grano", "lluvia", "luz", "delta_temp"]
CATEGORICAL_FEATURES = ["tipo_proceso"]

MIN_LOTES_TIEMPO = 5
MIN_LOTES_CALIDAD = 5

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def _a_esquema_produccion(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra humedad_grano_raw -> humedad_grano (mismo dato, solo el nombre que usan los
    notebooks de esta carpeta vs. el que usa el código de producción). No-op si ya viene con el
    nombre de producción."""
    if "humedad_grano_raw" in df.columns and "humedad_grano" not in df.columns:
        df = df.rename(columns={"humedad_grano_raw": "humedad_grano"})
    return df


def _rellenar_numericas(df: pd.DataFrame, medianas: dict) -> pd.DataFrame:
    df = df.copy()
    for col, mediana in medianas.items():
        df[col] = df[col].fillna(mediana)
    return df


def entrenar_isolation_forest(df_train: pd.DataFrame, contamination=None, n_estimators: int = 200) -> dict:
    """contamination=None usa la heurística original (tasa real de train, mínimo 2%) -- ver
    ML/09_evaluacion.ipynb y ML/10_ajuste_hiperparametros.ipynb para por qué ese mínimo fijo
    descalibra el modelo; pásale "auto" o un float explícito para usar el valor ya afinado."""
    df_train = _a_esquema_produccion(df_train)
    medianas = df_train[NUMERIC_FEATURES].median().to_dict()
    X = _rellenar_numericas(df_train, medianas)[NUMERIC_FEATURES]
    if contamination is None:
        contaminacion = float(np.clip(df_train["_es_anomalia"].mean(), 0.02, 0.3))
    else:
        contaminacion = contamination

    modelo = IsolationForest(contamination=contaminacion, random_state=42, n_estimators=n_estimators)
    modelo.fit(X)
    pred = modelo.predict(X)

    return {
        "modelo": modelo,
        "features": NUMERIC_FEATURES,
        "medianas_relleno": medianas,
        "contamination": contaminacion,
        "tasa_outliers_en_train": float((pred == -1).mean()),
        "n_filas_train": len(df_train),
    }


def entrenar_rf_tipo_anomalia(df_train: pd.DataFrame, n_estimators: int = 150, max_depth: int = 14) -> dict:
    df_train = _a_esquema_produccion(df_train)
    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    medianas = df_train[NUMERIC_FEATURES].median().to_dict()
    X = _rellenar_numericas(df_train, medianas)[cols]
    y = df_train["_tipo_anomalia"]

    if y.nunique() < 2:
        return {"omitido": f"solo hay {y.nunique()} clase(s) de _tipo_anomalia en train; hace "
                            "falta al menos 2 para entrenar un clasificador.",
                "n_filas_train": len(df_train)}

    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)], remainder="passthrough")
    pipe = Pipeline([
        ("prep", prep),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X, y)
    ajuste_train = float(pipe.score(X, y))  # diagnóstico de ajuste, NO evaluación de generalización

    return {
        "modelo": pipe,
        "features": cols,
        "medianas_relleno": medianas,
        "clases": sorted(y.unique().tolist()),
        "accuracy_ajuste_train": ajuste_train,
        "n_filas_train": len(df_train),
    }


def entrenar_ga_lluvia(
    df_train: pd.DataFrame, df_test: pd.DataFrame, horas: int = 3,
    generaciones: int = 80, tam_poblacion: int = 100, semilla: int = 42,
    metrica_fitness: str = "f1",
) -> dict:
    """Usa df_test únicamente para completar correctamente la etiqueta 'lluvia próxima' cerca
    del final de train (ver ga.etiquetar_train_test) -- el ajuste (evolución) en sí solo usa
    las filas de train, nunca features de test."""
    resultado = ga.entrenar_y_evaluar(
        df_train, df_test, horas=horas, generaciones=generaciones, tam_poblacion=tam_poblacion,
        semilla=semilla, metrica_fitness=metrica_fitness,
    )
    return {
        "mejor_individuo": resultado["mejor_individuo"],
        "escalador": resultado["escalador"],
        "features": ga.FEATURES_GA,
        "horas_anticipacion": horas,
        "metrica_fitness": metrica_fitness,
        "f1_ajuste_train": resultado["f1_train"],
        "n_filas_train": resultado["n_train"],
        "historial_fitness": resultado["historial_fitness"],
    }


def entrenar_regresor_tiempo(df_train: pd.DataFrame, min_lotes: int = MIN_LOTES_TIEMPO) -> dict:
    if "horas_restantes" not in df_train.columns:
        return {"omitido": "la columna horas_restantes no existe en este dataset."}
    etiquetado = df_train.dropna(subset=["horas_restantes"])
    n_lotes = etiquetado["id_lote"].nunique()
    if n_lotes < min_lotes:
        return {
            "omitido": f"solo {n_lotes} lote(s) en train con horas_restantes conocida (lote "
                       f"finalizado con fecha_fin_secado); se necesitan al menos {min_lotes}.",
            "n_lotes_disponibles": int(n_lotes),
        }
    # No se alcanza hoy con los datos reales del piloto; el entrenamiento real (RandomForestRegressor
    # + GroupShuffleSplit por lote) se implementa aquí cuando el guard anterior lo permita.
    raise NotImplementedError("Suficientes lotes para entrenar -- implementar RandomForestRegressor aquí.")


def entrenar_clasificador_calidad(df_train: pd.DataFrame, min_lotes: int = MIN_LOTES_CALIDAD) -> dict:
    if "_calidad_final_lote" not in df_train.columns:
        return {"omitido": "la columna _calidad_final_lote no existe en este dataset (la tabla "
                            "retroalimentacion_ml de Neon sigue vacía / sin migrar; ver migration.sql)."}
    etiquetado = df_train.dropna(subset=["_calidad_final_lote"])
    n_lotes = etiquetado["id_lote"].nunique()
    if n_lotes < min_lotes or etiquetado["_calidad_final_lote"].nunique() < 2:
        return {
            "omitido": f"solo {n_lotes} lote(s) en train con calidad_real conocida; se necesitan "
                       f"al menos {min_lotes} y >= 2 categorías distintas.",
            "n_lotes_disponibles": int(n_lotes),
        }
    raise NotImplementedError("Suficientes lotes para entrenar -- implementar RandomForestClassifier aquí.")


def main():
    here = os.path.dirname(__file__)
    train = pd.read_csv(os.path.join(here, "..", "data", "processed", "train.csv"), parse_dates=["timestamp"])
    test = pd.read_csv(os.path.join(here, "..", "data", "processed", "test.csv"), parse_dates=["timestamp"])

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    resumen = {}

    iso = entrenar_isolation_forest(train)
    joblib.dump(iso, os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
    resumen["isolation_forest"] = {k: v for k, v in iso.items() if k != "modelo"}
    print("IsolationForest entrenado:", resumen["isolation_forest"])

    rf_tipo = entrenar_rf_tipo_anomalia(train)
    if "modelo" in rf_tipo:
        joblib.dump(rf_tipo, os.path.join(ARTIFACTS_DIR, "rf_tipo_anomalia.joblib"))
    resumen["rf_tipo_anomalia"] = {k: v for k, v in rf_tipo.items() if k != "modelo"}
    print("RF tipo_anomalia:", resumen["rf_tipo_anomalia"])

    ga_lluvia = entrenar_ga_lluvia(train, test)
    joblib.dump(ga_lluvia, os.path.join(ARTIFACTS_DIR, "ga_lluvia.joblib"))
    resumen["ga_lluvia"] = {k: v for k, v in ga_lluvia.items() if k not in ("escalador", "historial_fitness")}
    print("AG lluvia:", resumen["ga_lluvia"])

    resumen["rf_tiempo_restante"] = entrenar_regresor_tiempo(train)
    print("RF tiempo_restante:", resumen["rf_tiempo_restante"])

    resumen["rf_calidad"] = entrenar_clasificador_calidad(train)
    print("RF calidad:", resumen["rf_calidad"])

    # Sello de tiempo + tamaño de train usado -- ML/monitoreo.py (paso 12) lo lee para saber
    # cuántas filas/días han pasado desde este entrenamiento y decidir si ya conviene reentrenar.
    resumen["fecha_entrenamiento"] = datetime.now(timezone.utc).isoformat()
    resumen["n_filas_train_total"] = len(train)

    with open(os.path.join(ARTIFACTS_DIR, "metricas_entrenamiento.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nArtefactos guardados en {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
