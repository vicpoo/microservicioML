"""
Motor de reglas de dominio para secado de café.

Basado en "Documento de Calidad del Café y Reglas del Dominio" (Cuadros 2, 3, 5, 9 y 10),
ajustado por definicion_problema_kajve.md (Sección 4.1, ya presente en este repo):

  - El hardware real usa BMP280, no BME280: NO existe humedad relativa ambiental como
    variable de entrada. Por eso `humedad_ambiental_alta` y `riesgo_moho_combinado`
    (que dependían de ella, además de viento) se eliminan por completo de v1, no solo
    se "simplifican". No hay anemómetro tampoco.
  - `lluvia` ya no es un float sintético normalizado 0-1: el firmware manda un booleano
    (`lluvia_detectada`) directo del FC-37. El preprocessor lo convierte a 1.0/0.0 antes
    de llegar aquí, así que el umbral LLUVIA_UMBRAL sigue funcionando sin cambios.
  - `humedad_grano` en la tabla real es el valor CRUDO del ADC del sensor capacitivo
    (smallint, típicamente 0-4095), no un porcentaje. Sin calibrar (ver
    RAW_GRANO_HUMEDO/RAW_GRANO_SECO más abajo), `secado_estancado` se omite en vez de
    disparar con una unidad que no es realmente un porcentaje.

Este módulo es la ÚNICA fuente de verdad para los umbrales: lo usan tanto el
generador del dataset real (scripts/recolectar_datos_reales.py) como el servicio de
inferencia en vivo (app/services/anomaly_detector.py), para que el modelo entrenado y
las reglas en producción nunca queden desalineados.
"""
from typing import Dict, List, Optional, Tuple

SEVERITY_ORDER = ["normal", "advertencia", "riesgo", "critico"]


def _rank(sev: str) -> int:
    return SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else 0


def peor_severidad(*severidades: str) -> str:
    return SEVERITY_ORDER[max(_rank(s) for s in severidades)] if severidades else "normal"


# --- Umbrales Cuadro 9 -------------------------------------------------

TEMP_GRANO_BANDAS = {
    "lavado": [(35, "normal"), (38, "advertencia"), (40, "riesgo"), (float("inf"), "critico")],
    "honey": [(35, "normal"), (38, "advertencia"), (40, "riesgo"), (float("inf"), "critico")],
    "natural": [(38, "normal"), (42, "advertencia"), (45, "riesgo"), (float("inf"), "critico")],
}

# Humedad de grano objetivo final: 10-12%. >12.5% almacenado = riesgo de reabsorción (Cuadro 3).
# OJO: estos valores son PORCENTAJE calibrado. El sensor real entrega un crudo del ADC
# (ver humedad_grano en app/models/lecturas_ambientales.py); hasta que se calibre en
# campo (dos puntos de referencia abajo), clasificar_secado_estancado se omite.
HUMEDAD_GRANO_MIN_SANO = 10.0
HUMEDAD_GRANO_MAX_ALMACEN = 12.5

# TODO(equipo kajve): llenar con los valores reales medidos en campo, ej:
# RAW_GRANO_HUMEDO = 1800  (raw con grano recién despulpado, ~50%)
# RAW_GRANO_SECO = 3100    (raw con grano seco, ~11%, verificado con medidor de referencia)
RAW_GRANO_HUMEDO: Optional[float] = None
RAW_GRANO_SECO: Optional[float] = None
HUMEDAD_HUMEDO_PCT = 50.0
HUMEDAD_SECO_PCT = 11.0

LUZ_INSUFICIENTE_LUX = 5000
LLUVIA_UMBRAL = 0.30  # 0.0/1.0 ya resuelto desde lluvia_detectada; cualquier 1.0 supera esto
FLUCTUACION_DELTA_C = 10.0  # °C entre 2 lecturas en <2h = "montaña rusa climática"
ESTANCAMIENTO_HORAS = 24
ESTANCAMIENTO_DELTA_MIN = 1.0  # % que debe bajar la humedad de grano en ESTANCAMIENTO_HORAS


def humedad_grano_calibrada() -> bool:
    return RAW_GRANO_HUMEDO is not None and RAW_GRANO_SECO is not None


def humedad_grano_raw_a_porcentaje(valor_crudo: Optional[float]) -> Optional[float]:
    """Interpola el valor crudo del ADC a % de humedad de grano. None si no está
    calibrado o si el valor crudo viene nulo (lectura faltante)."""
    if not humedad_grano_calibrada() or valor_crudo is None:
        return None
    if RAW_GRANO_SECO == RAW_GRANO_HUMEDO:
        return None
    proporcion = (valor_crudo - RAW_GRANO_HUMEDO) / (RAW_GRANO_SECO - RAW_GRANO_HUMEDO)
    porcentaje = HUMEDAD_HUMEDO_PCT + proporcion * (HUMEDAD_SECO_PCT - HUMEDAD_HUMEDO_PCT)
    return max(0.0, min(100.0, porcentaje))


def _banda(valor: float, bandas) -> str:
    for limite, etiqueta in bandas:
        if valor <= limite:
            return etiqueta
    return bandas[-1][1]


def clasificar_temperatura_grano(tipo_proceso: str, temp_grano: float) -> str:
    bandas = TEMP_GRANO_BANDAS.get(tipo_proceso, TEMP_GRANO_BANDAS["lavado"])
    return _banda(temp_grano, bandas)


def clasificar_lluvia(lluvia: float) -> Tuple[str, bool]:
    detectada = lluvia is not None and lluvia >= LLUVIA_UMBRAL
    return ("critico" if detectada else "normal"), detectada


def clasificar_fluctuacion(delta_temp_abs: Optional[float]) -> str:
    if delta_temp_abs is not None and delta_temp_abs >= FLUCTUACION_DELTA_C:
        return "advertencia"
    return "normal"


def clasificar_secado_estancado(delta_humedad_grano_24h_pct: Optional[float], humedad_grano_pct: Optional[float]) -> str:
    """Recibe valores YA CONVERTIDOS a porcentaje (ver humedad_grano_raw_a_porcentaje).
    Si el sensor no está calibrado, quien llama debe pasar None y esta función
    devuelve 'normal' (no evalúa) en vez de arriesgar un falso positivo/negativo."""
    if humedad_grano_pct is None or humedad_grano_pct <= HUMEDAD_GRANO_MIN_SANO:
        return "normal"
    if delta_humedad_grano_24h_pct is None:
        return "normal"
    if delta_humedad_grano_24h_pct < ESTANCAMIENTO_DELTA_MIN and humedad_grano_pct > HUMEDAD_GRANO_MAX_ALMACEN:
        return "riesgo"
    return "normal"


def clasificar_valor_imposible(temp_grano: Optional[float]) -> bool:
    """Filtro de cordura: descarta lecturas físicamente imposibles (glitch de sensor,
    ej. 181.6 °C ambiental observado en datos reales del piloto). Solo se valida
    temp_grano aquí: humedad_ambiental ya no es una variable del dominio (BMP280),
    y humedad_grano es un crudo de ADC cuyo rango "imposible" depende de la
    calibración de cada sensor, no de un 0-100 fijo."""
    if temp_grano is not None and (temp_grano < -10 or temp_grano > 85):
        return True
    return False


def evaluar_lectura(
    tipo_proceso: str,
    features: Dict[str, float],
    delta_temp_reciente: Optional[float] = None,
    delta_humedad_grano_24h_pct: Optional[float] = None,
) -> Dict:
    """Evalúa una lectura contra todas las reglas de dominio vigentes en v1.

    features esperadas: temperatura_grano, temperatura_ambiental, humedad_grano
    (crudo, informativo, no se usa directo en reglas), lluvia (0.0/1.0 ya resuelto
    desde lluvia_detectada), luz.
    delta_humedad_grano_24h_pct: ya debe venir convertido a % (o None si no calibrado).
    Devuelve severidad final + lista de alertas individuales + variables contribuyentes.
    """
    tipo_proceso = (tipo_proceso or "lavado").lower()
    alertas: List[Dict] = []

    if clasificar_valor_imposible(features.get("temperatura_grano")):
        alertas.append({
            "tipo": "valor_imposible",
            "severidad": "critico",
            "mensaje": "Lectura de sensor fuera de rango físico posible; revisar sensor/conexión.",
            "variable": "sensor",
        })

    sev_temp = clasificar_temperatura_grano(tipo_proceso, features.get("temperatura_grano", 0.0))
    if sev_temp != "normal":
        alertas.append({
            "tipo": "temperatura_alta",
            "severidad": sev_temp,
            "mensaje": "Temperatura del grano por encima del rango ideal para este proceso.",
            "variable": "temperatura_grano",
        })

    sev_lluvia, lluvia_detectada = clasificar_lluvia(features.get("lluvia", 0.0))
    if lluvia_detectada:
        alertas.append({
            "tipo": "lluvia_detectada",
            "severidad": sev_lluvia,
            "mensaje": "Lluvia detectada sobre el lote en secado.",
            "variable": "lluvia",
        })

    sev_fluct = clasificar_fluctuacion(delta_temp_reciente)
    if sev_fluct != "normal":
        alertas.append({
            "tipo": "fluctuacion_termica",
            "severidad": sev_fluct,
            "mensaje": "Cambio brusco de temperatura entre lecturas recientes.",
            "variable": "temperatura_grano",
        })

    humedad_grano_pct = humedad_grano_raw_a_porcentaje(features.get("humedad_grano"))
    sev_estancado = clasificar_secado_estancado(delta_humedad_grano_24h_pct, humedad_grano_pct)
    if sev_estancado != "normal":
        alertas.append({
            "tipo": "secado_estancado",
            "severidad": sev_estancado,
            "mensaje": "La humedad del grano no ha bajado en las últimas 24h.",
            "variable": "humedad_grano",
        })

    if (
        features.get("luz") is not None
        and features.get("luz") < LUZ_INSUFICIENTE_LUX
        and sev_estancado != "normal"
    ):
        alertas.append({
            "tipo": "radiacion_insuficiente",
            "severidad": "advertencia",
            "mensaje": "Poca luz solar sostenida mientras el secado está estancado.",
            "variable": "luz",
        })

    severidad_final = peor_severidad(*(a["severidad"] for a in alertas)) if alertas else "normal"
    variables = sorted({a["variable"] for a in alertas})
    tipo_principal = alertas[0]["tipo"] if alertas else "normal"
    if alertas:
        alertas.sort(key=lambda a: _rank(a["severidad"]), reverse=True)
        tipo_principal = alertas[0]["tipo"]

    return {
        "severidad": severidad_final,
        "es_anomalia": severidad_final != "normal",
        "alertas": alertas,
        "tipo_principal": tipo_principal,
        "variables_contribuyentes": variables,
    }


# --- Cuadro 10: recomendaciones por tipo de alerta ----------------------

RECOMENDACIONES: Dict[str, str] = {
    "temperatura_alta": "Mueve el lote a sombra parcial o reduce su exposición directa al sol hasta que la temperatura del grano baje al rango ideal.",
    "lluvia_detectada": "Prioridad máxima: cubre el lote con plástico o lona de inmediato para protegerlo de la lluvia.",
    "secado_estancado": "Revisa que el grosor de la capa esté entre 2 y 10 cm y aumenta la frecuencia de volteo; la humedad del grano no está bajando.",
    "fluctuacion_termica": "Se detectó un cambio brusco de temperatura ('montaña rusa climática'); aumenta la frecuencia de monitoreo preventivo.",
    "radiacion_insuficiente": "Poca luz solar sostenida con secado estancado: considera mover el lote a una zona con más exposición solar o usar secado asistido.",
    "valor_imposible": "Revisa la conexión y calibración del sensor; la última lectura está fuera de rango físico posible.",
    # patron_atipico_ml: lo agrega AnomalyDetector cuando el IsolationForest marca un outlier
    # que ninguna regla explícita cubrió (ver app/services/anomaly_detector.py). Antes de esto
    # no tenía entrada aquí y caía al texto de "normal" -- justo lo contrario de lo que
    # necesita alguien viendo una alerta real (hallazgo de ML/evaluacion.py::evaluar_cobertura_recomendaciones).
    "patron_atipico_ml": "El modelo detectó un patrón fuera de lo común no cubierto por las reglas explícitas: revisa el lote y las lecturas recientes con atención.",
    "normal": "El lote se encuentra dentro de los parámetros esperados para su tipo de proceso.",
}

# Título corto por tipo de alerta -- para notificaciones push (FCM) y cualquier vista que
# necesite un nombre de una línea en vez del texto completo de RECOMENDACIONES. Mismos tipos
# que RECOMENDACIONES/TIPO_SEVERIDAD_DEFAULT a propósito, para no mantener una tercera lista de
# tipos por separado.
TITULOS_CORTOS: Dict[str, str] = {
    "temperatura_alta": "Exceso de temperatura",
    "lluvia_detectada": "Lluvia detectada",
    "secado_estancado": "Secado estancado",
    "fluctuacion_termica": "Cambio brusco de temperatura",
    "radiacion_insuficiente": "Poca luz solar",
    "valor_imposible": "Sensor con error",
    "patron_atipico_ml": "Patrón atípico detectado",
    "riesgo_lluvia_proxima": "Riesgo de lluvia próxima",  # no es un tipo de rules.py: lo genera
                                                            # el AG (app/services/rain_predictor.py)
    "normal": "Todo en orden",
}


def recomendacion_para(tipo_alerta: str) -> str:
    return RECOMENDACIONES.get(tipo_alerta, RECOMENDACIONES["normal"])


def titulo_corto_para(tipo_alerta: str) -> str:
    return TITULOS_CORTOS.get(tipo_alerta, TITULOS_CORTOS["normal"])


# Severidad "por defecto" asociada a cada tipo de anomalía, usada cuando el
# clasificador RandomForest predice un tipo que la evaluación instantánea de
# reglas no marcó (p.ej. patrones que el modelo generalizó a partir del entrenamiento).
TIPO_SEVERIDAD_DEFAULT: Dict[str, str] = {
    "normal": "normal",
    "temperatura_alta": "riesgo",
    "lluvia_detectada": "critico",
    "secado_estancado": "riesgo",
    "fluctuacion_termica": "advertencia",
    "radiacion_insuficiente": "advertencia",
    "valor_imposible": "critico",
}


# --- Duración típica de proceso (Cuadro 1), en horas, para el predictor de tiempo restante ---
DURACION_HORAS = {
    "lavado": (6 * 24, 9 * 24),
    "honey": (8 * 24, 23 * 24),
    "natural": (10 * 24, 28 * 24),
}
