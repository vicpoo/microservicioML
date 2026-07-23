"""
ML/evaluacion.py

Paso 9 del pipeline de ML — Evaluación: mide qué tan bien generalizan (con
`data/processed/test.csv`, nunca visto durante el ajuste del paso 8) los modelos entrenados, y
los compara SIEMPRE contra un baseline simple para saber si de verdad aportan algo sobre "no
hacer nada" / "predecir lo obvio".

Métrica según el tipo de problema:
  - Clasificación (anomalías, tipo de anomalía, lluvia, calidad): accuracy, precision, recall,
    F1 y matriz de confusión.
  - Regresión (tiempo restante de secado): RMSE y MAE.

Baseline por salida (siempre calculado con estadísticas del TRAIN, nunca del test -- usar
estadísticas de test para el baseline sería la misma fuga de información que usar test para
entrenar):
  - Detección de anomalías / tipo de anomalía: "predecir siempre la clase mayoritaria de train"
    (equivalente a no tener ningún sistema de detección).
  - Lluvia: baseline de mayoría + baseline de "persistencia" (asumir que en H horas seguirá como
    está ahora mismo -- un baseline más fuerte y más justo para una señal casi-continua).
  - Tiempo restante: "predecir siempre la duración típica promedio del tipo de proceso" (Cuadro 1
    del documento de dominio), sin usar ninguna lectura de sensor.
  - Calidad final: "predecir siempre la clase de calidad más común entre los lotes ya reportados".

Los artefactos se cargan desde ML/artifacts/ (salida del paso 8).
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score,
)

from ML import prediccion_lluvia_ga as ga
from ML.entrenamiento import _a_esquema_produccion  # mismo renombre humedad_grano_raw -> humedad_grano

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Nombre "humedad_grano" (no "_raw") a propósito: debe coincidir con ML/entrenamiento.py, que a
# su vez coincide con lo que espera app/services/anomaly_detector.py / predictor.py en producción.
NUMERIC_FEATURES = ["temperatura_grano", "temperatura_ambiental", "humedad_grano", "lluvia", "luz", "delta_temp"]
CATEGORICAL_FEATURES = ["tipo_proceso"]

# Duración típica por proceso (Cuadro 1), para el baseline de tiempo restante -- mismos valores
# que ML/ingenieria_caracteristicas.py::DURACION_HORAS_PROMEDIO.
DURACION_HORAS_PROMEDIO = {
    "lavado": (6 * 24 + 9 * 24) / 2,
    "honey": (8 * 24 + 23 * 24) / 2,
    "natural": (10 * 24 + 28 * 24) / 2,
}


def cargar_artefactos(directorio: str = ARTIFACTS_DIR) -> dict:
    artefactos = {}
    for nombre in ["isolation_forest", "rf_tipo_anomalia", "ga_lluvia"]:
        ruta = os.path.join(directorio, f"{nombre}.joblib")
        artefactos[nombre] = joblib.load(ruta) if os.path.exists(ruta) else None
    return artefactos


# --- 1. Métricas genéricas --------------------------------------------------------------------

def evaluar_binario(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "matriz_confusion": cm.tolist(),
        "n": int(len(y_true)),
    }


def evaluar_multiclase(y_true, y_pred, labels=None) -> dict:
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "matriz_confusion": cm.tolist(),
        "labels": labels,
        "n": int(len(y_true)),
    }


def evaluar_regresion(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "n": int(len(y_true))}


# --- 2. Detección de anomalías (IsolationForest) ------------------------------------------------

def evaluar_isolation_forest(test_df: pd.DataFrame, artefacto: dict) -> dict:
    test_df = _a_esquema_produccion(test_df)
    X = test_df[artefacto["features"]].copy()
    for col, mediana in artefacto["medianas_relleno"].items():
        X[col] = X[col].fillna(mediana)
    y_true = test_df["_es_anomalia"].astype(int)

    pred_modelo = (artefacto["modelo"].predict(X) == -1).astype(int)
    pred_baseline = np.zeros(len(test_df), dtype=int)  # "nunca marcar nada raro"

    return {
        "modelo": evaluar_binario(y_true, pred_modelo),
        "baseline_siempre_normal": evaluar_binario(y_true, pred_baseline),
    }


# --- 3. Tipo de anomalía (RandomForestClassifier) -----------------------------------------------

def evaluar_rf_tipo_anomalia(train_df: pd.DataFrame, test_df: pd.DataFrame, artefacto: dict) -> dict:
    train_df = _a_esquema_produccion(train_df)
    test_df = _a_esquema_produccion(test_df)
    X = test_df[artefacto["features"]].copy()
    for col, mediana in artefacto["medianas_relleno"].items():
        X[col] = X[col].fillna(mediana)
    y_true = test_df["_tipo_anomalia"]

    pred_modelo = artefacto["modelo"].predict(X)

    clase_mayoritaria = train_df["_tipo_anomalia"].mode()[0]
    pred_baseline = [clase_mayoritaria] * len(test_df)

    labels = sorted(set(y_true) | set(pred_modelo) | {clase_mayoritaria})
    return {
        "modelo": evaluar_multiclase(y_true, pred_modelo, labels=labels),
        "baseline_clase_mayoritaria": evaluar_multiclase(y_true, pred_baseline, labels=labels),
        "clase_mayoritaria_usada": clase_mayoritaria,
    }


# --- 4. Predicción de lluvia (Algoritmo Genético) -----------------------------------------------

def evaluar_ga_lluvia(train_df: pd.DataFrame, test_df: pd.DataFrame, artefacto: dict, horas: int = 3) -> dict:
    train_et, test_et = ga.etiquetar_train_test(train_df, test_df, horas=horas)
    X_test, y_test = ga.preparar_X_y(test_et)
    X_train, y_train = ga.preparar_X_y(train_et)

    X_test_s = artefacto["escalador"].transform(X_test)
    individuo = np.array([artefacto["mejor_individuo"][f] for f in artefacto["features"]] + [artefacto["mejor_individuo"]["bias"]])
    pred_modelo = ga.predecir(individuo, X_test_s)

    clase_mayoritaria_train = int(round(y_train.mean())) if y_train.mean() >= 0.5 else 0
    pred_baseline_mayoria = np.full_like(y_test, clase_mayoritaria_train)

    # Baseline de persistencia: "en H horas seguirá como está ahora mismo" -- usa lluvia_sostenida
    # ACTUAL (no futura) de las mismas filas de test evaluadas, sin usar el AG para nada.
    datos_test_validos = test_et.dropna(subset=ga.FEATURES_GA + ["_lluvia_proxima"])
    pred_baseline_persistencia = datos_test_validos["lluvia_sostenida"].fillna(False).astype(int).to_numpy()

    return {
        "modelo": evaluar_binario(y_test, pred_modelo),
        "baseline_clase_mayoritaria": evaluar_binario(y_test, pred_baseline_mayoria),
        "baseline_persistencia": evaluar_binario(y_test, pred_baseline_persistencia),
        "horas_anticipacion": horas,
    }


# --- 5. Tiempo restante de secado (regresión) -- bloqueado hoy -----------------------------------

def evaluar_tiempo_restante(test_df: pd.DataFrame, min_lotes: int = 5) -> dict:
    etiquetado = test_df.dropna(subset=["horas_restantes"]) if "horas_restantes" in test_df.columns else test_df.iloc[0:0]
    n_lotes = etiquetado["id_lote"].nunique() if len(etiquetado) else 0
    if n_lotes < min_lotes:
        return {
            "omitido": f"solo {n_lotes} lote(s) en test con horas_restantes conocida; se "
                       f"necesitan al menos {min_lotes} lotes finalizados para evaluar con RMSE/MAE.",
            "baseline_propuesto": (
                "predecir siempre DURACION_HORAS_PROMEDIO[tipo_proceso] - horas_transcurridas "
                "(duración típica del Cuadro 1, sin usar ninguna lectura de sensor)"
            ),
        }
    duracion = etiquetado["tipo_proceso"].str.lower().map(DURACION_HORAS_PROMEDIO).fillna(DURACION_HORAS_PROMEDIO["lavado"])
    pred_baseline = (duracion - etiquetado["horas_transcurridas"]).clip(lower=0)
    return {"baseline_promedio_tipico": evaluar_regresion(etiquetado["horas_restantes"], pred_baseline)}


# --- 6. Calidad final (regresión, escala SCA 0-100) -- bloqueado hoy -----------------------------
# Antes era clasificación (4 categorías); ver migration.sql paso 10 y
# scripts/train_models.py::entrenar_regresor_calidad para la versión real que sí está
# implementada. Esta función es del pipeline offline de notebooks (ML/entrenamiento.py), que se
# deja igual de sin-implementar que antes -- solo se actualiza el criterio para reflejar que
# calidad_real ahora es un puntaje continuo, se evalúa con RMSE/MAE (evaluar_regresion), no con
# accuracy/f1.

def evaluar_calidad(test_df: pd.DataFrame, min_lotes: int = 5) -> dict:
    if "_calidad_final_lote" not in test_df.columns:
        return {"omitido": "la columna _calidad_final_lote no existe en este dataset "
                            "(retroalimentacion_ml sigue vacía en Neon)."}
    etiquetado = test_df.dropna(subset=["_calidad_final_lote"])
    n_lotes = etiquetado["id_lote"].nunique()
    if n_lotes < min_lotes:
        return {
            "omitido": f"solo {n_lotes} lote(s) en test con calidad_real conocida; se necesitan "
                       f"al menos {min_lotes}.",
            "baseline_propuesto": "predecir siempre el puntaje promedio de calidad_real entre los lotes ya reportados",
        }
    raise NotImplementedError("Suficientes lotes -- implementar evaluación real aquí con evaluar_regresion.")


# --- 7. Recomendaciones -- verificación de cobertura, no métricas de ML --------------------------

def evaluar_cobertura_recomendaciones(df: pd.DataFrame) -> dict:
    from app.services import rules
    tipos_reales = set(df["_tipo_anomalia"].dropna().unique())
    tipos_con_recomendacion = set(rules.RECOMENDACIONES.keys())
    faltantes = tipos_reales - tipos_con_recomendacion
    return {
        "tipos_de_anomalia_observados": sorted(tipos_reales),
        "cubiertos_por_rules_py": sorted(tipos_reales & tipos_con_recomendacion),
        "faltantes": sorted(faltantes),
        "cobertura_completa": len(faltantes) == 0,
    }
