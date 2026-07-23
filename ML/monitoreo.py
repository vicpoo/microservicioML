"""
ML/monitoreo.py

Paso 12 del pipeline de ML — Monitoreo y reentrenamiento: una vez el modelo está en
producción (paso 11), este módulo responde dos preguntas en curso:

  1. ¿Las predicciones que ya se hicieron en producción (tabla `predicciones`) se parecen
     a lo que después reportó el productor como resultado real (tabla `retroalimentacion_ml`)?
     -- monitoreo de DESEMPEÑO, reutiliza las mismas métricas (RMSE/MAE para tiempo Y calidad,
     ambas son regresión desde la migración a escala SCA 0-100) y el mismo baseline que
     ML/evaluacion.py (paso 9), pero contra datos EN VIVO en vez del test.csv offline.
  2. ¿Hay ya datos reales nuevos suficientes (lecturas nuevas en lecturas_ambientales, lotes
     finalizados en retroalimentacion_ml) como para que valga la pena reentrenar?
     -- monitoreo de DATOS, reutiliza los mismos umbrales MIN_LOTES_* que hoy bloquean
     entrenar_regresor_tiempo/entrenar_regresor_calidad en ML/entrenamiento.py.

Este módulo NO reentrena solo -- solo diagnostica y recomienda. Reentrenar automáticamente
sin revisión humana sería peligroso: el paso 10 ya mostró que un modelo puede "ganar" en
validación y fallar en datos reales (ver ML/artifacts/metricas_paso10_ajuste.json), así que
cada reentrenamiento real debe seguir siendo manual: correr 08_entrenamiento.ipynb ->
09_evaluacion.ipynb -> 10_ajuste_hiperparametros.ipynb, revisar el reporte, y solo entonces
copiar los artefactos nuevos a app/ml/artifacts/ (igual que se hizo a mano en el paso 11).

Se usa desde dos lugares:
  - scripts/monitorear_modelos.py -- pensado para correr por cron en el servidor real
    (este módulo no se puede "programar" desde aquí, Cowork no tiene acceso al servidor
    donde vive el microservicio desplegado).
  - GET /internal/monitoreo/salud -- para verlo en caliente vía API (Gestor/ops).
"""
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.alertas import Alerta
from app.models.lecturas_ambientales import LecturaAmbiental
from app.models.predicciones import Prediccion
from app.models.retroalimentacion_ml import RetroalimentacionML

from ML.entrenamiento import ARTIFACTS_DIR, MIN_LOTES_CALIDAD, MIN_LOTES_TIEMPO
from ML.evaluacion import DURACION_HORAS_PROMEDIO, evaluar_regresion

METRICAS_ENTRENAMIENTO_PATH = os.path.join(ARTIFACTS_DIR, "metricas_entrenamiento.json")

# Cuántas filas nuevas en lecturas_ambientales hace falta acumular desde el último
# entrenamiento antes de sugerir reentrenar IsolationForest/RF/AG (que sí tienen datos hoy,
# a diferencia de tiempo/calidad, que siguen bloqueados por MIN_LOTES_*). Es una heurística de
# arranque -- ajústala cuando tengas más lotes reales y una idea mejor de cada cuánto conviene.
UMBRAL_FILAS_NUEVAS_REENTRENAMIENTO = 2000

# Ventana usada para comparar tasa de alertas reciente vs anterior (detección simple de drift
# de comportamiento, no de datos).
VENTANA_ALERTAS_DIAS_DEFAULT = 7


def _cargar_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- 1. Desempeño en producción: predicciones vs retroalimentación real ------------------------

def evaluar_desempeno_produccion(db: Session) -> dict:
    """Compara, para cada lote con retroalimentación real, la ÚLTIMA predicción que el
    servicio hizo para ese lote contra lo que el productor reportó al final. Es la versión "en
    vivo" de ML/evaluacion.py: misma idea (comparar contra un baseline simple), pero con datos
    de producción reales en vez del test.csv offline."""
    retro = db.query(RetroalimentacionML).all()
    if not retro:
        return {"omitido": "retroalimentacion_ml está vacía todavía; no hay resultados reales "
                            "contra qué comparar las predicciones."}

    filas = []
    for r in retro:
        ultima_pred = (
            db.query(Prediccion)
            .filter(Prediccion.id_lote == r.id_lote)
            .order_by(Prediccion.fecha_prediccion.desc())
            .first()
        )
        if ultima_pred is None:
            continue
        filas.append({
            "id_lote": r.id_lote,
            "tipo_proceso": r.tipo_proceso,
            "tiempo_real_horas": float(r.tiempo_real_horas),
            "tiempo_estimado_horas": float(ultima_pred.tiempo_estimado_horas) if ultima_pred.tiempo_estimado_horas is not None else None,
            "calidad_real": r.calidad_real,
            "calidad_estimada": ultima_pred.calidad_estimada,
        })

    if not filas:
        return {"omitido": f"hay {len(retro)} lote(s) con retroalimentación real, pero ninguno "
                            "tiene una fila en predicciones asociada todavía."}

    df = pd.DataFrame(filas)
    resultado = {"n_lotes_con_retroalimentacion": len(retro), "n_lotes_comparados": len(df)}

    tiempo = df.dropna(subset=["tiempo_estimado_horas"])
    if len(tiempo) >= 1:
        duracion_tipica = (
            tiempo["tipo_proceso"].str.lower().map(DURACION_HORAS_PROMEDIO)
            .fillna(DURACION_HORAS_PROMEDIO["lavado"])
        )
        resultado["tiempo_restante"] = {
            "modelo": evaluar_regresion(tiempo["tiempo_real_horas"], tiempo["tiempo_estimado_horas"]),
            "baseline_promedio_tipico": evaluar_regresion(tiempo["tiempo_real_horas"], duracion_tipica),
        }
    else:
        resultado["tiempo_restante"] = {
            "omitido": "ninguna predicción de tiempo disponible (rf_tiempo_restante.joblib "
                       "sigue sin poder entrenarse con datos reales)."
        }

    # calidad_real/calidad_estimada ahora son un puntaje continuo (escala SCA 0-100), no una
    # categoría -- ya no tiene sentido "accuracy" (igualdad exacta de string); se evalúa con
    # RMSE/MAE, igual que tiempo_restante, vía evaluar_regresion. Ver migration.sql paso 10.
    calidad = df.dropna(subset=["calidad_estimada", "calidad_real"])
    if len(calidad) >= 1:
        resultado["calidad"] = evaluar_regresion(calidad["calidad_real"], calidad["calidad_estimada"])
    else:
        resultado["calidad"] = {
            "omitido": "ninguna predicción de calidad disponible (rf_calidad.joblib sigue sin "
                       "poder entrenarse con datos reales), o calidad_real todavía no se ha "
                       "reportado para ningún lote (llega vía POST /internal/lotes/{id}/catacion)."
        }

    return resultado


# --- 2. Monitoreo de comportamiento: tasa de alertas en el tiempo ------------------------------

def tasa_alertas(db: Session, dias: int = VENTANA_ALERTAS_DIAS_DEFAULT) -> dict:
    """Compara la cantidad de alertas (riesgo/crítico) de los últimos `dias` contra la ventana
    anterior de la misma duración. Un salto grande puede ser una anomalía real del proceso
    (el modelo haciendo su trabajo) o un síntoma de drift/sensor descalibrado -- este número
    por sí solo no distingue cuál es, pero avisa que hay que revisar con calma."""
    ahora = datetime.now(timezone.utc)
    inicio_actual = ahora - timedelta(days=dias)
    inicio_anterior = inicio_actual - timedelta(days=dias)

    actual = db.query(func.count(Alerta.id_alerta)).filter(Alerta.fecha_generada >= inicio_actual).scalar() or 0
    anterior = (
        db.query(func.count(Alerta.id_alerta))
        .filter(Alerta.fecha_generada >= inicio_anterior, Alerta.fecha_generada < inicio_actual)
        .scalar() or 0
    )

    if anterior > 0:
        razon = round(actual / anterior, 2)
    elif actual > 0:
        razon = None  # "infinito" -- no había alertas antes y ahora sí; se marca aparte
    else:
        razon = 1.0  # sin alertas en ninguna ventana, no hay salto que reportar

    return {
        "ventana_dias": dias,
        "alertas_ventana_actual": int(actual),
        "alertas_ventana_anterior": int(anterior),
        "razon_actual_vs_anterior": razon,
        "posible_drift": bool((razon is not None and razon >= 2.0) or (razon is None and actual >= 3)),
    }


# --- 3. ¿Hay datos nuevos suficientes para reentrenar? ------------------------------------------

def disponibilidad_datos_nuevos(db: Session) -> dict:
    """Compara cuántas filas/lotes reales hay HOY contra lo que se usó la última vez que se
    entrenó (guardado en ML/artifacts/metricas_entrenamiento.json por ML/entrenamiento.py), y
    contra los umbrales MIN_LOTES_* que hoy bloquean entrenar_regresor_tiempo/
    entrenar_clasificador_calidad."""
    metricas_previas = _cargar_json(METRICAS_ENTRENAMIENTO_PATH) or {}
    n_filas_train_previo = metricas_previas.get(
        "n_filas_train_total", (metricas_previas.get("isolation_forest") or {}).get("n_filas_train", 0)
    )
    fecha_entrenamiento = metricas_previas.get("fecha_entrenamiento")

    n_filas_actual = db.query(func.count(LecturaAmbiental.id_lectura)).scalar() or 0
    n_lotes_retro = db.query(func.count(func.distinct(RetroalimentacionML.id_lote))).scalar() or 0

    return {
        "fecha_ultimo_entrenamiento": fecha_entrenamiento,
        "n_filas_train_ultimo_entrenamiento": int(n_filas_train_previo),
        "n_filas_lecturas_ambientales_hoy": int(n_filas_actual),
        "filas_nuevas_desde_ultimo_entrenamiento": int(max(n_filas_actual - n_filas_train_previo, 0)),
        "umbral_filas_nuevas": UMBRAL_FILAS_NUEVAS_REENTRENAMIENTO,
        "n_lotes_con_retroalimentacion_real": int(n_lotes_retro),
        "lotes_faltantes_para_tiempo_restante": max(MIN_LOTES_TIEMPO - n_lotes_retro, 0),
        "lotes_faltantes_para_calidad": max(MIN_LOTES_CALIDAD - n_lotes_retro, 0),
    }


# --- 4. Decisión: ¿conviene reentrenar? (diagnóstico, NUNCA automático) -------------------------

def necesita_reentrenamiento(db: Session) -> dict:
    datos = disponibilidad_datos_nuevos(db)
    desempeno = evaluar_desempeno_produccion(db)
    razones = []

    if datos["filas_nuevas_desde_ultimo_entrenamiento"] >= UMBRAL_FILAS_NUEVAS_REENTRENAMIENTO:
        razones.append(
            f"Hay {datos['filas_nuevas_desde_ultimo_entrenamiento']} lecturas nuevas desde el "
            f"último entrenamiento (umbral: {UMBRAL_FILAS_NUEVAS_REENTRENAMIENTO}) -- vale la "
            "pena re-correr 08_entrenamiento.ipynb / 10_ajuste_hiperparametros.ipynb con el "
            "dataset ampliado."
        )
    if datos["lotes_faltantes_para_tiempo_restante"] == 0:
        razones.append(
            f"Ya hay {datos['n_lotes_con_retroalimentacion_real']} lote(s) con retroalimentación "
            f"real (>= {MIN_LOTES_TIEMPO}) -- entrenar_regresor_tiempo ya no debería omitirse: "
            "implementa el RandomForestRegressor pendiente en ML/entrenamiento.py y córrelo."
        )
    if datos["lotes_faltantes_para_calidad"] == 0:
        razones.append(
            f"Ya hay {datos['n_lotes_con_retroalimentacion_real']} lote(s) con retroalimentación "
            f"real (>= {MIN_LOTES_CALIDAD}) -- entrenar_regresor_calidad ya no debería "
            "omitirse: implementa el RandomForestRegressor pendiente en ML/entrenamiento.py y "
            "córrelo."
        )
    tiempo_prod = desempeno.get("tiempo_restante") if isinstance(desempeno, dict) else None
    if isinstance(tiempo_prod, dict) and "modelo" in tiempo_prod:
        rmse_modelo = tiempo_prod["modelo"]["rmse"]
        rmse_baseline = tiempo_prod["baseline_promedio_tipico"]["rmse"]
        if rmse_modelo > rmse_baseline:
            razones.append(
                f"El modelo de tiempo restante en producción tiene peor RMSE en vivo "
                f"({rmse_modelo:.1f}h) que el baseline simple ({rmse_baseline:.1f}h) -- no está "
                "aportando, revisar antes de seguir usándolo."
            )

    return {
        "necesita_reentrenamiento": len(razones) > 0,
        "razones": razones,
        "disponibilidad_datos": datos,
        "desempeno_produccion": desempeno,
    }


def resumen_salud(db: Session, dias_alertas: int = VENTANA_ALERTAS_DIAS_DEFAULT) -> dict:
    """Un solo reporte con todo lo anterior -- lo que expone GET /internal/monitoreo/salud y
    lo que imprime scripts/monitorear_modelos.py."""
    return {
        "fecha_reporte": datetime.now(timezone.utc).isoformat(),
        "reentrenamiento": necesita_reentrenamiento(db),
        "monitoreo_alertas": tasa_alertas(db, dias=dias_alertas),
    }
