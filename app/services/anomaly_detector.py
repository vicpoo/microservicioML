import joblib
import os
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
import numpy as np


class AnomalyDetector:
    def __init__(self, model_path: str = "app/ml/artifacts/isolation_forest.joblib"):
        self.model_path = model_path
        self.model = None
        self.expected_features = 10
        self._load_or_train_default()

    def _load_or_train_default(self) -> None:
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            if getattr(self.model, "n_features_in_", None) != self.expected_features:
                self.model = self._train_default_model()
            return

        self.model = self._train_default_model()

    def _train_default_model(self):
        X = np.array([
            [28, 28, 60, 45, 4, 0, 30000, 0, 15, 0],
            [29, 29, 62, 46, 4.2, 0, 31000, 0, 14.8, 0],
            [30, 30, 58, 44, 4.5, 0, 32000, 0, 12.9, 0],
            [32, 31, 85, 48, 0.4, 0, 10000, 1, 212.5, 0],
            [33, 32, 90, 50, 0.2, 1, 8000, 1, 450, 1],
            [42, 33, 70, 46, 2.2, 0, 25000, 9, 31.8, 0],
        ], dtype=float)
        model = IsolationForest(contamination=0.2, random_state=42)
        model.fit(X)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        return model

    def predict(self, features: Dict[str, float]) -> Tuple[bool, float, List[str]]:
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
        row = np.array([
            features.get("temperatura_grano", 0.0),
            features.get("temperatura_ambiental", 0.0),
            features.get("humedad_ambiental", 0.0),
            features.get("humedad_grano", 0.0),
            features.get("viento", 0.0),
            features.get("lluvia", 0.0),
            features.get("luz", 0.0),
            features.get("delta_temp", 0.0),
            features.get("indice_moho", 0.0),
            features.get("lluvia_binaria", 0.0),
        ], dtype=float).reshape(1, -1)
        pred = self.model.predict(row)[0]
        score = float(self.model.decision_function(row)[0])
        is_anomaly = pred == -1
        contrib = []
        if is_anomaly:
            if features.get("humedad_ambiental", 0.0) > 80:
                contrib.append("humedad_ambiental")
            if features.get("viento", 0.0) < 1:
                contrib.append("viento")
            if features.get("lluvia", 0.0) >= 0.5:
                contrib.append("lluvia")
        return is_anomaly, score, contrib
