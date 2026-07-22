"""
ML/ajuste_hiperparametros.py

Paso 10 del pipeline de ML — Ajuste de hiperparámetros.

Metodología: para no "espiar" test durante el ajuste (harían falta dos vueltas de mirar test:
una para elegir hiperparámetros y otra para la evaluación final, lo cual invalida la segunda),
este módulo separa TRAIN (paso 6) en dos partes usando el mismo mecanismo de división temporal
del paso 6 (`ML/division_datos.py`):

  - `train_fit`  (80% más antiguo de train) -- para ajustar cada candidato.
  - `train_val`  (20% más reciente de train) -- para puntuar candidatos y elegir el ganador.

`test.csv` (paso 6) se toca solo UNA vez al final, para confirmar el resultado con la
configuración ganadora ya fija -- igual que en el paso 9, nunca antes.

Se afinan los 3 modelos que sí tienen datos suficientes hoy:
  - IsolationForest: `contamination`.
  - RandomForestClassifier (tipo de anomalía): `n_estimators`, `max_depth`.
  - Algoritmo Genético (lluvia): métrica de fitness (`f1` vs `f1_macro`) e hiperparámetros de la
    evolución (generaciones, tamaño de población, escala de mutación).
"""
import os

import numpy as np
import pandas as pd

from ML import division_datos as dd
from ML import entrenamiento as ent
from ML import evaluacion as ev
from ML import prediccion_lluvia_ga as ga

VAL_TEST_SIZE = 0.2
VAL_RANDOM_STATE = 42


def dividir_train_val(train: pd.DataFrame) -> tuple:
    train_fit, train_val, info = dd.dividir_datos(train, test_size=VAL_TEST_SIZE, random_state=VAL_RANDOM_STATE)
    return train_fit, train_val, info


# --- 1. IsolationForest: contamination -----------------------------------------------------------

def ajustar_isolation_forest(train_fit: pd.DataFrame, train_val: pd.DataFrame, candidatos=None) -> pd.DataFrame:
    candidatos = candidatos if candidatos is not None else ["auto", 0.005, 0.01, 0.02, 0.05]
    filas = []
    for contam in candidatos:
        artefacto = ent.entrenar_isolation_forest(train_fit, contamination=contam)
        resultado = ev.evaluar_isolation_forest(train_val, artefacto)
        filas.append({"contamination": contam, **resultado["modelo"]})
    return pd.DataFrame(filas).set_index("contamination")[["accuracy", "precision", "recall", "f1"]]


# --- 2. RandomForestClassifier (tipo de anomalía): n_estimators, max_depth -----------------------

def ajustar_rf_tipo_anomalia(train_fit: pd.DataFrame, train_val: pd.DataFrame, grid=None) -> pd.DataFrame:
    grid = grid if grid is not None else [
        {"n_estimators": 100, "max_depth": 8},
        {"n_estimators": 150, "max_depth": 14},
        {"n_estimators": 200, "max_depth": 20},
        {"n_estimators": 300, "max_depth": None},
    ]
    filas = []
    for params in grid:
        artefacto = ent.entrenar_rf_tipo_anomalia(train_fit, **params)
        if "modelo" not in artefacto:
            filas.append({**params, "omitido": artefacto.get("omitido")})
            continue
        resultado = ev.evaluar_rf_tipo_anomalia(train_fit, train_val, artefacto)
        filas.append({**params, "accuracy": resultado["modelo"]["accuracy"], "f1_macro": resultado["modelo"]["f1_macro"]})
    return pd.DataFrame(filas)


# --- 3. Algoritmo Genético (lluvia): métrica de fitness + hiperparámetros de evolución -----------

def ajustar_ga_lluvia(
    train_fit: pd.DataFrame, train_val: pd.DataFrame, test: pd.DataFrame, configs=None, horas: int = 3,
) -> pd.DataFrame:
    """`test` se pasa solo para completar correctamente la etiqueta 'lluvia próxima' cerca del
    final de train_val (mismo motivo que en ga.etiquetar_train_test) -- el ajuste y el puntaje
    de cada candidato usan EXCLUSIVAMENTE train_fit/train_val, test no se toca para nada más
    aquí."""
    configs = configs if configs is not None else [
        {"metrica_fitness": "f1", "generaciones": 60, "tam_poblacion": 80, "escala_mutacion": 0.3},
        {"metrica_fitness": "f1_macro", "generaciones": 60, "tam_poblacion": 80, "escala_mutacion": 0.3},
        {"metrica_fitness": "f1_macro", "generaciones": 100, "tam_poblacion": 150, "escala_mutacion": 0.1},
    ]
    etiquetadas = ga.etiquetar_particiones(
        {"train_fit": train_fit, "train_val": train_val, "test": test}, horas=horas,
    )
    fit_et, val_et = etiquetadas["train_fit"], etiquetadas["train_val"]
    X_fit, y_fit = ga.preparar_X_y(fit_et)
    X_val, y_val = ga.preparar_X_y(val_et)

    escalador = ga.EscaladorMinMax().fit(X_fit)
    X_fit_s = escalador.transform(X_fit)
    X_val_s = escalador.transform(X_val)

    filas = []
    for cfg in configs:
        resultado = ga.evolucionar(
            X_fit_s, y_fit, generaciones=cfg["generaciones"], tam_poblacion=cfg["tam_poblacion"],
            escala_mutacion=cfg.get("escala_mutacion", 0.3), metrica_fitness=cfg["metrica_fitness"],
        )
        pred_val = ga.predecir(resultado.mejor_individuo, X_val_s)
        pct_positivo = float(np.mean(pred_val)) if len(pred_val) else float("nan")
        filas.append({
            **cfg,
            "n_val_usable": len(y_val),
            "f1_val": ga._f1(y_val, pred_val) if len(y_val) else float("nan"),
            "f1_macro_val": ga._f1_macro(y_val, pred_val) if len(y_val) else float("nan"),
            "pct_predicho_positivo_val": pct_positivo,
        })
    return pd.DataFrame(filas)
