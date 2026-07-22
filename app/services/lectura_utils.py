#app/services/lectura_utils.py
"""Construcción del vector de 6 features a partir de una fila real de lecturas_ambientales,
y cálculo de horas_transcurridas desde el inicio de secado de un lote. Antes vivía duplicado
inline en app/api/routes/internal.py; ahora también lo usa app/services/poller.py (paso 12,
revisión periódica) -- se factoriza aquí para que ambos caminos construyan el mismo vector de
la misma forma, en vez de mantener dos copias que se puedan desincronizar."""
from datetime import datetime, timezone
from typing import Dict

from app.models.lecturas_ambientales import LecturaAmbiental
from app.models.lotes_cafe import LoteCafe


def construir_features(lectura: LecturaAmbiental) -> Dict[str, float]:
    # humedad_ambiental ya no existe: el hardware real es BMP280, no BME280. "lluvia" se
    # resuelve del booleano lluvia_detectada que ya manda el firmware, no de un float
    # sintético. humedad_grano se pasa CRUDO (sin calibrar); rules.py lo convierte a % si hay
    # calibración disponible.
    features = {
        "temperatura_grano": float(lectura.temperatura_grano) if lectura.temperatura_grano is not None else 0.0,
        "temperatura_ambiental": float(lectura.temperatura) if lectura.temperatura is not None else 0.0,
        "humedad_grano": float(lectura.humedad_grano) if lectura.humedad_grano is not None else 0.0,
        "lluvia": 1.0 if lectura.lluvia_detectada else 0.0,
        "luz": float(lectura.luz) if lectura.luz is not None else 0.0,
    }
    features["delta_temp"] = features["temperatura_grano"] - features["temperatura_ambiental"]
    return features


def calcular_horas_transcurridas(lote: LoteCafe) -> float:
    if not lote.fecha_inicio_secado:
        return 0.0
    inicio = lote.fecha_inicio_secado
    if inicio.tzinfo is None:
        inicio = inicio.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - inicio).total_seconds() / 3600.0, 0.0)
