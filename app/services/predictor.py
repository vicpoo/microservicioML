#app/services/predictor.py
import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts")


class Predictor:
    """Predicciones de negocio: horas restantes de secado y calidad final estimada (puntaje
    escala SCA 0-100 -- una aproximación basada en condiciones de secado, no una catación real).

    Ambas son RandomForest, entrenados exclusivamente con datos reales de Neon
    (scripts/recolectar_datos_reales.py + scripts/train_models.py; el dataset sintético
    quedó deprecado). Mientras no haya suficientes lotes reales finalizados para entrenarlos
    (ver ML/README.md, paso 12), `rf_tiempo.joblib`/`rf_calidad.joblib` simplemente no existen
    en `app/ml/artifacts/` y `_load()` devuelve None -- `predecir()` responde con
    tiempo_estimado_horas/calidad_estimada en None en vez de usar un artefacto viejo o inventado.
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.rf_tiempo = self._load("rf_tiempo_restante.joblib")
        self.rf_calidad = self._load("rf_calidad.joblib")

    def _load(self, filename: str) -> Optional[dict]:
        path = os.path.join(self.artifacts_dir, filename)
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    def _fila(self, tipo_proceso: str, features: Dict[str, float], horas_transcurridas: float) -> pd.DataFrame:
        return pd.DataFrame([{
            "tipo_proceso": tipo_proceso,
            "temperatura_grano": features["temperatura_grano"],
            "temperatura_ambiental": features["temperatura_ambiental"],
            "humedad_grano": features["humedad_grano"],
            "lluvia": features["lluvia"],
            "luz": features["luz"],
            "delta_temp": features["delta_temp"],
            "horas_transcurridas": horas_transcurridas,
        }])

    def predecir(self, tipo_proceso: str, features: Dict[str, float], horas_transcurridas: float = 0.0) -> Dict:
        tipo_proceso = (tipo_proceso or "lavado").lower()
        fila = self._fila(tipo_proceso, features, horas_transcurridas)

        # Cada predicción se envuelve en su propio try/except: un artefacto viejo/incompatible
        # (p. ej. entrenado con un esquema de columnas anterior) no debe tumbar TODA la
        # respuesta de /detect -- las demás salidas (anomalías, recomendaciones) siguen siendo
        # útiles aunque esta predicción en particular no esté disponible. Se loguea para que el
        # equipo note que hay que reentrenar/reemplazar ese artefacto, en vez de fallar en
        # silencio de forma permanente.
        tiempo_estimado_horas = None
        if self.rf_tiempo is not None:
            try:
                pred = float(self.rf_tiempo["modelo"].predict(fila)[0])
                tiempo_estimado_horas = round(max(pred, 0.0), 1)
            except Exception as exc:
                logger.warning(f"[predictor] rf_tiempo_restante.joblib incompatible/con error, se omite tiempo_estimado_horas: {exc}")

        calidad_estimada, confianza_calidad = None, None
        if self.rf_calidad is not None:
            try:
                pipe = self.rf_calidad["modelo"]
                # rf_calidad ahora es un RandomForestRegressor (puntaje SCA 0-100), no un
                # clasificador de 4 categorías -- ver migration.sql paso 10 y
                # scripts/train_models.py::entrenar_regresor_calidad. calidad_estimada es una
                # aproximación indirecta basada en condiciones de secado, NO una catación real.
                pred = float(pipe.predict(fila)[0])
                calidad_estimada = round(min(max(pred, 0.0), 100.0), 1)

                # "confianza" ya no es la probabilidad de una clase (eso era propio del
                # clasificador); aquí es una heurística basada en qué tan de acuerdo están los
                # árboles del bosque entre sí: se mide la desviación estándar de sus predicciones
                # individuales y se convierte a una escala 0-100 donde una dispersión de 25 puntos
                # o más (un cuarto del rango completo de la escala SCA) se considera confianza
                # nula. No es una probabilidad calibrada, es solo una señal relativa de qué tan
                # "seguro" está el modelo de este caso en particular.
                prep = pipe.named_steps["prep"]
                reg = pipe.named_steps["reg"]
                X_prep = prep.transform(fila)
                preds_arboles = np.array([arbol.predict(X_prep)[0] for arbol in reg.estimators_])
                dispersion = float(preds_arboles.std())
                confianza_calidad = round(max(0.0, 100.0 - (dispersion / 25.0) * 100.0), 1)
            except Exception as exc:
                logger.warning(f"[predictor] rf_calidad.joblib incompatible/con error, se omite calidad_estimada: {exc}")

        return {
            "tiempo_estimado_horas": tiempo_estimado_horas,
            "calidad_estimada": calidad_estimada,
            "confianza": confianza_calidad,
        }
