#scripts/train_models.py
"""
Entrena los 4 artefactos del pipeline de ML a partir de data/raw/lecturas_ml_training.csv
(generado por scripts/generar_dataset.py) y los guarda en app/ml/artifacts/.

Artefactos:
  1. isolation_forest.joblib   -> IsolationForest, detección de outliers no supervisada
  2. rf_tipo_anomalia.joblib   -> RandomForestClassifier, predice tipo de anomalía (incluye "normal")
  3. rf_tiempo_restante.joblib -> RandomForestRegressor, horas restantes de secado
  4. rf_calidad.joblib         -> RandomForestClassifier, calidad final estimada

Cada artefacto es un sklearn Pipeline completo (incluye el one-hot de tipo_proceso), así el
servicio solo arma un DataFrame de una fila con las columnas crudas y llama .predict().
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(__file__)
RAW_CSV = os.path.join(HERE, "..", "data", "raw", "lecturas_ml_training.csv")
RETRO_CSV = os.path.join(HERE, "..", "data", "raw", "retroalimentacion_real.csv")
PROCESSED_CSV = os.path.join(HERE, "..", "data", "processed", "lecturas_limpias.csv")
ARTIFACTS_DIR = os.path.join(HERE, "..", "app", "ml", "artifacts")

NUMERIC_FEATURES = [
    "temperatura_grano",
    "temperatura_ambiental",
    "humedad_ambiental",
    "humedad_grano",
    "lluvia",
    "luz",
    "delta_temp",
]
CATEGORICAL_FEATURES = ["tipo_proceso"]


def cargar_y_limpiar() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)

    # RNF-19: si hay resultados reales reportados por productores (ver
    # scripts/exportar_retroalimentacion.py), se combinan con el dataset sintético.
    if os.path.exists(RETRO_CSV):
        df_real = pd.read_csv(RETRO_CSV)
    else:
        df_real = pd.DataFrame()
    if not df_real.empty:
        print(f"Combinando {len(df_real)} lecturas reales de retroalimentacion_ml (RNF-19) con el dataset sintético.")
        df = pd.concat([df, df_real], ignore_index=True)
    else:
        print("Sin datos reales de retroalimentación todavía (RNF-19); entrenando solo con dataset sintético. "
              "Corre scripts/exportar_retroalimentacion.py cuando haya lotes finalizados reportados por productores.")

    df = df.dropna(subset=["tipo_proceso", "id_lote"])
    for col in ["temperatura_grano", "temperatura_ambiental", "humedad_ambiental", "humedad_grano", "lluvia", "luz"]:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
    df["delta_temp"] = df["temperatura_grano"] - df["temperatura_ambiental"]
    os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    return df


def _preprocesador() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )


def entrenar_isolation_forest(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES]
    contamination = float(np.clip(df["_es_anomalia"].mean(), 0.02, 0.3))
    modelo = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    modelo.fit(X)
    pred = modelo.predict(X)
    tasa_outliers = (pred == -1).mean()
    joblib.dump({"modelo": modelo, "features": NUMERIC_FEATURES}, os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
    return {"contamination": contamination, "tasa_outliers_detectados": round(float(tasa_outliers), 4)}


def entrenar_clasificador_tipo(df: pd.DataFrame):
    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[cols]
    y = df["_tipo_anomalia"]
    groups = df["id_lote"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    pipe = Pipeline([
        ("prep", _preprocesador()),
        ("clf", RandomForestClassifier(
            n_estimators=150, max_depth=14, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
    ])
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = pipe.predict(X.iloc[test_idx])
    metricas = {
        "accuracy": round(float(accuracy_score(y.iloc[test_idx], y_pred)), 4),
        "f1_macro": round(float(f1_score(y.iloc[test_idx], y_pred, average="macro")), 4),
    }
    # Reentrena con todos los datos para el artefacto final de producción
    pipe.fit(X, y)
    joblib.dump({"modelo": pipe, "features": cols, "clases": list(pipe.named_steps["clf"].classes_)},
                os.path.join(ARTIFACTS_DIR, "rf_tipo_anomalia.joblib"))
    return metricas


def entrenar_regresor_tiempo(df: pd.DataFrame):
    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["horas_transcurridas"]
    X = df[cols]
    y = df["horas_restantes"]
    groups = df["id_lote"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    pipe = Pipeline([
        ("prep", _preprocesador()),
        ("reg", RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = pipe.predict(X.iloc[test_idx])
    rmse = float(np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred)))
    mae = float(mean_absolute_error(y.iloc[test_idx], y_pred))
    metricas = {"rmse_horas": round(rmse, 2), "mae_horas": round(mae, 2)}

    pipe.fit(X, y)
    joblib.dump({"modelo": pipe, "features": cols}, os.path.join(ARTIFACTS_DIR, "rf_tiempo_restante.joblib"))
    return metricas


def entrenar_clasificador_calidad(df: pd.DataFrame):
    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["horas_transcurridas"]
    X = df[cols]
    y = df["_calidad_final_lote"].astype(str)
    groups = df["id_lote"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    pipe = Pipeline([
        ("prep", _preprocesador()),
        ("clf", RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
    ])
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = pipe.predict(X.iloc[test_idx])
    metricas = {
        "accuracy": round(float(accuracy_score(y.iloc[test_idx], y_pred)), 4),
        "f1_macro": round(float(f1_score(y.iloc[test_idx], y_pred, average="macro")), 4),
    }

    pipe.fit(X, y)
    joblib.dump({"modelo": pipe, "features": cols, "clases": list(pipe.named_steps["clf"].classes_)},
                os.path.join(ARTIFACTS_DIR, "rf_calidad.joblib"))
    return metricas


def main():
    import sys
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = cargar_y_limpiar()
    print(f"Filas de entrenamiento (limpias): {len(df):,}")

    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    metricas_path = os.path.join(ARTIFACTS_DIR, "metricas.json")
    resultados = json.load(open(metricas_path, encoding="utf-8")) if os.path.exists(metricas_path) else {}

    if only in ("all", "isolation_forest"):
        resultados["isolation_forest"] = entrenar_isolation_forest(df)
        print("IsolationForest:", resultados["isolation_forest"])
    if only in ("all", "tipo"):
        resultados["rf_tipo_anomalia"] = entrenar_clasificador_tipo(df)
        print("RF tipo_anomalia (holdout por lote):", resultados["rf_tipo_anomalia"])
    if only in ("all", "tiempo"):
        resultados["rf_tiempo_restante"] = entrenar_regresor_tiempo(df)
        print("RF tiempo_restante (holdout por lote):", resultados["rf_tiempo_restante"])
    if only in ("all", "calidad"):
        resultados["rf_calidad"] = entrenar_clasificador_calidad(df)
        print("RF calidad (holdout por lote):", resultados["rf_calidad"])

    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nArtefactos guardados en {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
