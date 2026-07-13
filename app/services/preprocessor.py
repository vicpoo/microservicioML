#app/services/preprocessor.py
from typing import Dict


class Preprocessor:
    """Arma el vector de 6 variables reales (sin viento, no hay anemómetro en el kit IoT)."""

    feature_names = [
        "temperatura_grano",     # DS18B20
        "temperatura_ambiental",  # BME280
        "humedad_ambiental",      # BME280
        "humedad_grano",          # sensor capacitivo de humedad de grano
        "lluvia",                 # FC-37, normalizado 0-1
        "luz",                    # BH1750, lux
    ]

    def transform(self, payload: Dict[str, float]) -> Dict[str, float]:
        values = {name: float(payload.get(name, 0.0)) for name in self.feature_names}
        values["delta_temp"] = values["temperatura_grano"] - values["temperatura_ambiental"]
        return values
