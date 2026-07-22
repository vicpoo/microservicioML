#app/services/rain_predictor.py
"""
Conecta en vivo el Algoritmo Genético de predicción de lluvia (paso 7 del pipeline, entrenado
en ML/prediccion_lluvia_ga.py / ML/entrenamiento.py::entrenar_ga_lluvia). Hasta ahora
ga_lluvia.joblib solo se evaluaba offline (notebooks 07-10) -- nunca se llamaba desde /detect
ni desde el webhook/poller. Esto es lo que le faltaba para de verdad "hacer con los datos del
sensor un AG para la predicción de lluvia" en producción, no solo en los notebooks.

Qué predice: riesgo de que llueva en las próximas H horas (no si está lloviendo AHORA -- eso
ya lo cubre el sensor FC-37 vía app/services/rules.py, sin necesidad de ML).

Features (ML/prediccion_lluvia_ga.py::FEATURES_GA): presion_hpa, luz (vienen de la lectura
actual) + lluvia_eventos_24h y horas_desde_ultima_lluvia (HISTÓRICAS, se calculan aquí con una
consulta a lecturas_ambientales del mismo lote -- a diferencia de las 6 features estándar de
anomaly_detector.py/predictor.py, que solo miran la lectura actual).

Caveat documentado (ver ML/README.md): el modelo se entrenó usando `lluvia_sostenida` (columna
derivada SOLO offline en el paso 4, que filtra el parpadeo conocido del sensor FC-37). En
producción todavía no existe una versión "sostenida" en vivo de esa señal (mismo hallazgo
abierto que bloquea a rules.py::clasificar_lluvia()), así que aquí se usa lluvia_detectada
cruda -- el propio ML/ingenieria_caracteristicas.py ya documenta este fallback y advierte que
puede sobreestimar eventos de lluvia. Mientras eso no se resuelva, el riesgo que devuelve esto
puede pecar de alarmista, no de omiso -- razonable para una alerta preventiva (mejor cubrir el
lote de más que de menos).
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import joblib
import numpy as np
from sqlalchemy.orm import Session

from app.models.lecturas_ambientales import LecturaAmbiental

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts")
VENTANA_EVENTOS_24H = timedelta(hours=24)
SIN_LLUVIA_PREVIA_HORAS = 999.0  # el AG no vio NaN en train; usar un número grande en vez de
                                  # NaN para no romper el score lineal cuando el lote nunca vio lluvia


class PredictorLluvia:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artefacto: Optional[dict] = self._load(artifacts_dir)

    def _load(self, artifacts_dir: str) -> Optional[dict]:
        path = os.path.join(artifacts_dir, "ga_lluvia.joblib")
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    def _features_historicas(self, db: Session, id_lote: int, ahora_naive_utc: datetime) -> Dict[str, float]:
        """Misma definición que ML/ingenieria_caracteristicas.py, pero calculada contra el
        historial real del lote en lecturas_ambientales (con lluvia_detectada cruda -- ver
        docstring del módulo). `ahora_naive_utc` debe venir SIN tzinfo (mismo formato que
        guarda la columna timestamp), para no mezclar naive/aware al comparar."""
        ultima_lluvia = (
            db.query(LecturaAmbiental.timestamp)
            .filter(LecturaAmbiental.id_lote == id_lote, LecturaAmbiental.lluvia_detectada.is_(True))
            .order_by(LecturaAmbiental.timestamp.desc())
            .first()
        )
        if ultima_lluvia is None or ultima_lluvia[0] is None:
            horas_desde_ultima_lluvia = SIN_LLUVIA_PREVIA_HORAS
        else:
            horas_desde_ultima_lluvia = max(
                (ahora_naive_utc - ultima_lluvia[0]).total_seconds() / 3600.0, 0.0
            )

        eventos_24h = (
            db.query(LecturaAmbiental)
            .filter(
                LecturaAmbiental.id_lote == id_lote,
                LecturaAmbiental.lluvia_detectada.is_(True),
                LecturaAmbiental.timestamp >= ahora_naive_utc - VENTANA_EVENTOS_24H,
                LecturaAmbiental.timestamp <= ahora_naive_utc,
            )
            .count()
        )
        return {
            "horas_desde_ultima_lluvia": horas_desde_ultima_lluvia,
            "lluvia_eventos_24h": float(eventos_24h),
        }

    def predecir(
        self, db: Session, id_lote: int, presion_hpa: Optional[float], luz: float,
        ahora: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """None si no hay artefacto, si falta presión (el BMP280 real siempre la manda; solo
        faltaría en un /detect manual sin ese campo), o si algo en el artefacto no es
        compatible -- nunca lanza excepción, mismo patrón defensivo que predictor.py/
        anomaly_detector.py: esta predicción es un extra, no debe tumbar el resto de /detect."""
        if self.artefacto is None or presion_hpa is None:
            return None
        # lecturas_ambientales.timestamp no guarda zona horaria (naive, asumido UTC -- mismo
        # criterio que inference.py::_contexto_lote en otros cálculos de horas). Se normaliza
        # `ahora` a naive-UTC aquí, una sola vez, para no mezclar aware/naive en las consultas.
        ahora = ahora or datetime.now(timezone.utc)
        ahora_naive_utc = ahora.replace(tzinfo=None) if ahora.tzinfo is not None else ahora
        try:
            historicas = self._features_historicas(db, id_lote, ahora_naive_utc)
            x = np.array([[
                presion_hpa,
                luz,
                historicas["lluvia_eventos_24h"],
                historicas["horas_desde_ultima_lluvia"],
            ]], dtype=float)
            x_escalado = self.artefacto["escalador"].transform(x)

            individuo = np.array(
                [self.artefacto["mejor_individuo"][f] for f in self.artefacto["features"]]
                + [self.artefacto["mejor_individuo"]["bias"]]
            )
            pesos, bias = individuo[:-1], individuo[-1]
            score = float((x_escalado @ pesos)[0] + bias)

            return {
                "riesgo_lluvia_proxima": bool(score > 0),
                "horas_anticipacion": int(self.artefacto.get("horas_anticipacion", 3)),
                "score": round(score, 4),
            }
        except Exception as exc:
            logger.warning(f"[rain_predictor] ga_lluvia.joblib incompatible/con error, se omite: {exc}")
            return None
