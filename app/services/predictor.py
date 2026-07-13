#app/services/predictor.py
import os
from typing import Dict, Optional

import joblib
import pandas as pd

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts")


class Predictor:
    """Predicciones de negocio: horas restantes de secado y calidad final estimada.

    Ambas son RandomForest entrenados sobre el dataset sintético (scripts/generar_dataset.py).
    La predicción de calidad a partir de una sola lectura es una estimación de tendencia
    ("si las condiciones actuales se mantienen"), no un veredicto definitivo; su exactitud
    mejorará entrenando con datos reales de lotes ya finalizados.
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
            "humedad_ambiental": features["humedad_ambiental"],
            "humedad_grano": features["humedad_grano"],
            "lluvia": features["lluvia"],
            "luz": features["luz"],
            "delta_temp": features["delta_temp"],
            "horas_transcurridas": horas_transcurridas,
        }])

    def predecir(self, tipo_proceso: str, features: Dict[str, float], horas_transcurridas: float = 0.0) -> Dict:
        tipo_proceso = (tipo_proceso or "lavado").lower()
        fila = self._fila(tipo_proceso, features, horas_transcurridas)

        tiempo_estimado_horas = None
        if self.rf_tiempo is not None:
            pred = float(self.rf_tiempo["modelo"].predict(fila)[0])
            tiempo_estimado_horas = round(max(pred, 0.0), 1)

        calidad_estimada, confianza_calidad = None, None
        if self.rf_calidad is not None:
            pipe = self.rf_calidad["modelo"]
            proba = pipe.predict_proba(fila)[0]
            clases = pipe.named_steps["clf"].classes_
            idx = proba.argmax()
            calidad_estimada = str(clases[idx])
            confianza_calidad = round(float(proba[idx]) * 100, 1)

        return {
            "tiempo_estimado_horas": tiempo_estimado_horas,
            "calidad_estimada": calidad_estimada,
            "confianza": confianza_calidad,
        }
