#app/services/preprocessor.py
import math
from typing import Dict


class Preprocessor:
    def __init__(self):
        self.feature_names = [
            "temperatura_grano",
            "temperatura_ambiental",
            "humedad_ambiental",
            "humedad_grano",
            "viento",
            "lluvia",
            "luz",
        ]

    def transform(self, payload: Dict[str, float]) -> Dict[str, float]:
        values = {}
        for name in self.feature_names:
            values[name] = float(payload.get(name, 0.0))

        values["delta_temp"] = values["temperatura_grano"] - values["temperatura_ambiental"]
        values["indice_moho"] = values["humedad_ambiental"] / max(values["viento"], 0.1)
        values["lluvia_binaria"] = 1.0 if values["lluvia"] >= 0.5 else 0.0
        return values
