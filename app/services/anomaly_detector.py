# Archivo: app/services/anomaly_detector.py
# Carpeta: microservicioMLL/app/services/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

import logging
import os
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd

from app.services.rules import TIPO_SEVERIDAD_DEFAULT, evaluar_lectura, peor_severidad

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts")


class AnomalyDetector:
    """Ensemble: motor de reglas de dominio (determinista) + RandomForest (supervisado,
    generaliza patrones) + IsolationForest (no supervisado, atrapa outliers novedosos que
    ni las reglas ni el clasificador conocen). Las reglas garantizan que situaciones
    críticas conocidas (ej. lluvia) SIEMPRE se detecten aunque el modelo aún no las haya
    visto; el ensemble ML generaliza más allá de los umbrales fijos.
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.rf_tipo = self._load("rf_tipo_anomalia.joblib")
        self.isolation = self._load("isolation_forest.joblib")

    def _load(self, filename: str) -> Optional[dict]:
        path = os.path.join(self.artifacts_dir, filename)
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    def _fila_ml(self, tipo_proceso: str, features: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([{
            "tipo_proceso": tipo_proceso,
            "temperatura_grano": features["temperatura_grano"],
            "temperatura_ambiental": features["temperatura_ambiental"],
            "humedad_grano": features["humedad_grano"],
            "lluvia": features["lluvia"],
            "luz": features["luz"],
            "delta_temp": features["delta_temp"],
        }])

    def predict(
        self,
        tipo_proceso: str,
        features: Dict[str, float],
        delta_temp_reciente: Optional[float] = None,
        delta_humedad_grano_24h_pct: Optional[float] = None,
    ) -> Dict:
        tipo_proceso = (tipo_proceso or "lavado").lower()

        regla = evaluar_lectura(
            tipo_proceso, features,
            delta_temp_reciente=delta_temp_reciente,
            delta_humedad_grano_24h_pct=delta_humedad_grano_24h_pct,
        )

        fila = self._fila_ml(tipo_proceso, features)

        # Igual que en predictor.py: un artefacto viejo/incompatible no debe tumbar la
        # detección completa -- las reglas de dominio (evaluar_lectura, arriba) siguen
        # funcionando solas si el RandomForest/IsolationForest fallan por cualquier motivo.
        tipo_ml, confianza_ml = "normal", 0.0
        if self.rf_tipo is not None:
            try:
                pipe = self.rf_tipo["modelo"]
                proba = pipe.predict_proba(fila)[0]
                clases = pipe.named_steps["clf"].classes_
                idx = proba.argmax()
                tipo_ml, confianza_ml = clases[idx], float(proba[idx])
            except Exception as exc:
                logger.warning(f"[anomaly_detector] rf_tipo_anomalia.joblib incompatible/con error, se usan solo las reglas: {exc}")

        outlier_ml = False
        score_if = 0.0
        if self.isolation is not None:
            try:
                modelo_if = self.isolation["modelo"]
                cols = self.isolation["features"]
                fila_if = fila[cols]
                outlier_ml = bool(modelo_if.predict(fila_if)[0] == -1)
                score_if = float(modelo_if.decision_function(fila_if)[0])
            except Exception as exc:
                logger.warning(f"[anomaly_detector] isolation_forest.joblib incompatible/con error, se omite el score de outlier: {exc}")

        # Las reglas (umbrales exactos del PDF) son la fuente autoritativa cuando SÍ detectan
        # algo: dan una severidad graduada y precisa. El RandomForest solo puede ELEVAR la
        # severidad cuando las reglas no encontraron nada pero el modelo reconoce un patrón
        # de anomalía (generalización más allá de los umbrales fijos); no debe "aplanar" a
        # su severidad por defecto un caso que las reglas ya clasificaron más fino.
        severidad_ml = TIPO_SEVERIDAD_DEFAULT.get(tipo_ml, "normal")
        if regla["severidad"] == "normal" and tipo_ml != "normal":
            severidad_final = peor_severidad(regla["severidad"], severidad_ml)
        else:
            severidad_final = regla["severidad"]

        variables = set(regla["variables_contribuyentes"])
        alertas = list(regla["alertas"])
        if outlier_ml and not regla["alertas"]:
            variables.add("patron_atipico_ml")
            alertas.append({
                "tipo": "patron_atipico_ml",
                "severidad": "advertencia",
                "mensaje": "El modelo detectó un patrón fuera de lo común no cubierto por las reglas explícitas.",
                "variable": "patron_atipico_ml",
            })
            severidad_final = peor_severidad(severidad_final, "advertencia")

        es_anomalia = severidad_final != "normal" or outlier_ml
        tipo_principal = regla["tipo_principal"] if regla["tipo_principal"] != "normal" else tipo_ml

        return {
            "es_anomalia": es_anomalia,
            "severidad": severidad_final,
            "tipo_principal": tipo_principal,
            "variables_contribuyentes": sorted(variables) or ["sin_datos"],
            "alertas": alertas,
            "score_isolation_forest": round(score_if, 4),
            "confianza_ml": round(confianza_ml * 100, 1),
        }