#scripts/train_models.py
"""
Entrena los 4 artefactos del pipeline de ML a partir de datos REALES recolectados con
scripts/recolectar_datos_reales.py (ya no del dataset sintético; ver
scripts/generar_dataset.py, deprecado) y los guarda en app/ml/artifacts/.

Artefactos:
  1. isolation_forest.joblib   -> IsolationForest, detección de outliers no supervisada
  2. rf_tipo_anomalia.joblib   -> RandomForestClassifier, predice tipo de anomalía (incluye "normal")
  3. rf_tiempo_restante.joblib -> RandomForestRegressor, horas restantes de secado
  4. rf_calidad.joblib         -> RandomForestRegressor, puntaje de calidad estimado (escala SCA
                                  0-100; ya no es un clasificador de 4 categorías -- ver
                                  Documento de Calidad del Café, Sección 7, y migration.sql paso 10)

Cada artefacto es un sklearn Pipeline completo (incluye el one-hot de tipo_proceso), así el
servicio solo arma un DataFrame de una fila con las columnas crudas y llama .predict().

IMPORTANTE: con pocos lotes reales (ver definicion_problema_kajve.md, Sección 6), es
normal y ESPERADO que rf_tiempo_restante y/o rf_calidad no tengan suficientes filas
etiquetadas para entrenar todavía. Este script lo detecta y salta esos modelos con un
aviso claro, en vez de entrenar con 2-3 ejemplos y producir un modelo que parece
"funcionar" pero no generaliza nada.
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
REAL_CSV = os.path.join(HERE, "..", "data", "raw", "lecturas_reales_entrenamiento.csv")
PROCESSED_CSV = os.path.join(HERE, "..", "data", "processed", "lecturas_limpias.csv")
ARTIFACTS_DIR = os.path.join(HERE, "..", "app", "ml", "artifacts")

# Nota: ya no incluye humedad_ambiental (el hardware real es BMP280, no BME280; ver
# definicion_problema_kajve.md Sección 4.1). "lluvia" es 0.0/1.0 resuelto desde
# lluvia_detectada, no un float sintético de intensidad.
NUMERIC_FEATURES = [
    "temperatura_grano",
    "temperatura_ambiental",
    "humedad_grano",
    "lluvia",
    "luz",
    "delta_temp",
]
CATEGORICAL_FEATURES = ["tipo_proceso"]

# Umbrales mínimos para considerar que vale la pena entrenar cada modelo supervisado
# (no aplica al IsolationForest ni al clasificador de tipo de anomalía, que pueden
# aprovechar cualquier lectura, tenga o no lote finalizado).
MIN_LOTES_TIEMPO = 5
MIN_LOTES_CALIDAD = 5


def cargar_y_limpiar() -> pd.DataFrame:
    if not os.path.exists(REAL_CSV):
        raise FileNotFoundError(
            f"No existe {REAL_CSV}. Corre primero: python scripts/recolectar_datos_reales.py"
        )
    df = pd.read_csv(REAL_CSV)
    if df.empty:
        raise ValueError(
            "El dataset real está vacío (0 lecturas en lecturas_ambientales). "
            "No hay nada que entrenar todavía."
        )

    df = df.dropna(subset=["tipo_proceso", "id_lote"])
    for col in ["temperatura_grano", "temperatura_ambiental", "humedad_grano", "lluvia", "luz"]:
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
    return {"contamination": contamination, "tasa_outliers_detectados": round(float(tasa_outliers), 4), "n_filas": len(df)}


def entrenar_clasificador_tipo(df: pd.DataFrame):
    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[cols]
    y = df["_tipo_anomalia"]
    groups = df["id_lote"]

    if df["id_lote"].nunique() < 2 or y.nunique() < 2:
        return {"omitido": "menos de 2 lotes o menos de 2 clases de tipo_anomalia; no se puede "
                            "hacer un split por lote todavía", "n_filas": len(df)}

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
        "n_filas": len(df),
    }
    pipe.fit(X, y)
    joblib.dump({"modelo": pipe, "features": cols, "clases": list(pipe.named_steps["clf"].classes_)},
                os.path.join(ARTIFACTS_DIR, "rf_tipo_anomalia.joblib"))
    return metricas


def entrenar_regresor_tiempo(df: pd.DataFrame):
    df_etiquetado = df.dropna(subset=["horas_restantes"])
    n_lotes = df_etiquetado["id_lote"].nunique()
    if n_lotes < MIN_LOTES_TIEMPO:
        return {
            "omitido": f"solo {n_lotes} lote(s) con horas_restantes conocida (fecha_fin_secado "
                       f"registrada); se necesitan al menos {MIN_LOTES_TIEMPO} para un split "
                       "por lote razonable. Usa la heurística por proceso mientras tanto "
                       "(ver definicion_problema_kajve.md, Sección 3.2).",
            "n_lotes_disponibles": int(n_lotes),
        }

    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["horas_transcurridas"]
    df_etiquetado = df_etiquetado.dropna(subset=["horas_transcurridas"])
    X = df_etiquetado[cols]
    y = df_etiquetado["horas_restantes"]
    groups = df_etiquetado["id_lote"]

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
    metricas = {"rmse_horas": round(rmse, 2), "mae_horas": round(mae, 2), "n_lotes": int(n_lotes)}

    pipe.fit(X, y)
    joblib.dump({"modelo": pipe, "features": cols}, os.path.join(ARTIFACTS_DIR, "rf_tiempo_restante.joblib"))
    return metricas


def entrenar_regresor_calidad(df: pd.DataFrame):
    """RandomForestRegressor sobre calidad_real (escala SCA 0-100). Antes era un clasificador de
    4 categorías (excelente/buena/regular/baja); ver migration.sql paso 10 para el detalle de la
    migración. Mismo patrón que entrenar_regresor_tiempo() -- ya no hace falta el chequeo de "al
    menos 2 categorías distintas" que tenía el clasificador, un valor continuo no tiene ese
    problema."""
    df_etiquetado = df.dropna(subset=["_calidad_final_lote"])
    n_lotes = df_etiquetado["id_lote"].nunique()
    if n_lotes < MIN_LOTES_CALIDAD:
        return {
            "omitido": f"solo {n_lotes} lote(s) con calidad_real conocida "
                       f"(retroalimentacion_ml); se necesitan al menos {MIN_LOTES_CALIDAD} "
                       "para un split por lote razonable. Usa el criterio basado en "
                       "historial de alertas mientras tanto (ver definicion_problema_kajve.md, "
                       "Sección 3.3).",
            "n_lotes_disponibles": int(n_lotes),
        }

    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["horas_transcurridas"]
    df_etiquetado = df_etiquetado.dropna(subset=["horas_transcurridas"])
    X = df_etiquetado[cols]
    y = df_etiquetado["_calidad_final_lote"].astype(float)
    groups = df_etiquetado["id_lote"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    pipe = Pipeline([
        ("prep", _preprocesador()),
        ("reg", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = pipe.predict(X.iloc[test_idx])
    rmse = float(np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred)))
    mae = float(mean_absolute_error(y.iloc[test_idx], y_pred))
    metricas = {"rmse_puntos": round(rmse, 2), "mae_puntos": round(mae, 2), "n_lotes": int(n_lotes)}

    pipe.fit(X, y)
    joblib.dump({"modelo": pipe, "features": cols}, os.path.join(ARTIFACTS_DIR, "rf_calidad.joblib"))
    return metricas


def main():
    import sys
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = cargar_y_limpiar()
    print(f"Filas de entrenamiento (reales, limpias): {len(df):,}")

    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    metricas_path = os.path.join(ARTIFACTS_DIR, "metricas.json")
    resultados = json.load(open(metricas_path, encoding="utf-8")) if os.path.exists(metricas_path) else {}

    if only in ("all", "isolation_forest"):
        resultados["isolation_forest"] = entrenar_isolation_forest(df)
        print("IsolationForest:", resultados["isolation_forest"])
    if only in ("all", "tipo"):
        resultados["rf_tipo_anomalia"] = entrenar_clasificador_tipo(df)
        print("RF tipo_anomalia:", resultados["rf_tipo_anomalia"])
    if only in ("all", "tiempo"):
        resultados["rf_tiempo_restante"] = entrenar_regresor_tiempo(df)
        print("RF tiempo_restante:", resultados["rf_tiempo_restante"])
    if only in ("all", "calidad"):
        resultados["rf_calidad"] = entrenar_regresor_calidad(df)
        print("RF calidad:", resultados["rf_calidad"])

    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nArtefactos y metricas guardados en {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
