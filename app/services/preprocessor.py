#app/services/preprocessor.py
from typing import Dict


class Preprocessor:
    """Arma el vector de 5 variables reales (sin viento -no hay anemómetro- y sin
    humedad ambiental -el BMP280 real no la mide, a diferencia del BME280 que se
    había asumido originalmente; ver definicion_problema_kajve.md Sección 4.1)."""

    feature_names = [
        "temperatura_grano",     # DS18B20
        "temperatura_ambiental",  # BMP280
        "humedad_grano",          # sensor capacitivo (crudo de ADC si no está calibrado)
        "lluvia",                 # FC-37: 1.0/0.0 resuelto desde lluvia_detectada
        "luz",                    # BH1750, lux
    ]

    def transform(self, payload: Dict[str, float]) -> Dict[str, float]:
        values = {name: float(payload.get(name, 0.0)) for name in self.feature_names}
        values["delta_temp"] = values["temperatura_grano"] - values["temperatura_ambiental"]
        return values
