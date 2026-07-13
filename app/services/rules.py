#app/services/rules.py
"""
Motor de reglas de dominio para secado de café.

Basado en "Documento de Calidad del Café y Reglas del Dominio" (Cuadros 2, 3, 5, 9 y 10):
valores validados contra estándares SCA/CQI + literatura Cenicafé/PDG, pendientes de
validación en campo con un Q Grader (así lo indica el documento fuente).

Este módulo es la ÚNICA fuente de verdad para los umbrales: lo usan tanto el generador
del dataset sintético (scripts/generar_dataset.py) como el servicio de inferencia en vivo
(app/services/anomaly_detector.py), para que el modelo entrenado y las reglas en
producción nunca queden desalineados.

NOTA: no hay anemómetro en el kit de sensores IoT real (BME280, DS18B20, BH1750, FC-37,
sensor de humedad de grano), así que "viento" se eliminó por completo del dominio de
variables. Las reglas combinadas que el documento ligaba a viento (riesgo_moho_combinado)
se simplifican a humedad_ambiental + temperatura sostenidas.
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

HUMEDAD_AMBIENTAL_BANDAS = [
    (65, "normal"),
    (80, "advertencia"),
    (90, "riesgo"),
    (float("inf"), "critico"),
]

# Humedad de grano objetivo final: 10-12%. >12.5% almacenado = riesgo de reabsorción (Cuadro 3).
HUMEDAD_GRANO_MIN_SANO = 10.0
HUMEDAD_GRANO_MAX_ALMACEN = 12.5

LUZ_INSUFICIENTE_LUX = 5000
LLUVIA_UMBRAL = 0.30  # normalizado 0-1; FC-37 por encima de esto = lluvia detectada
FLUCTUACION_DELTA_C = 10.0  # °C entre 2 lecturas en <2h = "montaña rusa climática"
ESTANCAMIENTO_HORAS = 24
ESTANCAMIENTO_DELTA_MIN = 1.0  # % que debe bajar la humedad de grano en ESTANCAMIENTO_HORAS


def _banda(valor: float, bandas) -> str:
    for limite, etiqueta in bandas:
        if valor <= limite:
            return etiqueta
    return bandas[-1][1]


def clasificar_temperatura_grano(tipo_proceso: str, temp_grano: float) -> str:
    bandas = TEMP_GRANO_BANDAS.get(tipo_proceso, TEMP_GRANO_BANDAS["lavado"])
    return _banda(temp_grano, bandas)


def clasificar_humedad_ambiental(humedad_ambiental: float) -> str:
    return _banda(humedad_ambiental, HUMEDAD_AMBIENTAL_BANDAS)


def clasificar_lluvia(lluvia: float) -> Tuple[str, bool]:
    detectada = lluvia is not None and lluvia >= LLUVIA_UMBRAL
    return ("critico" if detectada else "normal"), detectada


def clasificar_fluctuacion(delta_temp_abs: Optional[float]) -> str:
    if delta_temp_abs is not None and delta_temp_abs >= FLUCTUACION_DELTA_C:
        return "advertencia"
    return "normal"


def clasificar_secado_estancado(delta_humedad_grano_24h: Optional[float], humedad_grano: Optional[float]) -> str:
    """delta_humedad_grano_24h: cuánto bajó (positivo) la humedad de grano en ~24h."""
    if humedad_grano is None or humedad_grano <= HUMEDAD_GRANO_MIN_SANO:
        return "normal"
    if delta_humedad_grano_24h is None:
        return "normal"
    if delta_humedad_grano_24h < ESTANCAMIENTO_DELTA_MIN and humedad_grano > HUMEDAD_GRANO_MAX_ALMACEN:
        return "riesgo"
    return "normal"


def clasificar_riesgo_moho(humedad_ambiental: float, temp_grano: float) -> str:
    """Simplificación de 'riesgo_moho_combinado' sin dato de viento: HR alta + temp sostenida."""
    if humedad_ambiental is not None and humedad_ambiental > 80 and temp_grano is not None and temp_grano > 25:
        return "riesgo"
    return "normal"


def clasificar_valor_imposible(temp_grano: float, humedad_ambiental: float, humedad_grano: float) -> bool:
    if temp_grano is not None and (temp_grano < 0 or temp_grano > 85):
        return True
    if humedad_ambiental is not None and (humedad_ambiental < 0 or humedad_ambiental > 100):
        return True
    if humedad_grano is not None and (humedad_grano < 0 or humedad_grano > 100):
        return True
    return False


def evaluar_lectura(
    tipo_proceso: str,
    features: Dict[str, float],
    delta_temp_reciente: Optional[float] = None,
    delta_humedad_grano_24h: Optional[float] = None,
) -> Dict:
    """Evalúa una lectura contra todas las reglas de dominio.

    features esperadas: temperatura_grano, temperatura_ambiental, humedad_ambiental,
    humedad_grano, lluvia, luz.
    Devuelve severidad final + lista de alertas individuales + variables contribuyentes.
    """
    tipo_proceso = (tipo_proceso or "lavado").lower()
    alertas: List[Dict] = []

    if clasificar_valor_imposible(
        features.get("temperatura_grano"), features.get("humedad_ambiental"), features.get("humedad_grano")
    ):
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

    sev_hr = clasificar_humedad_ambiental(features.get("humedad_ambiental", 0.0))
    if sev_hr != "normal":
        alertas.append({
            "tipo": "humedad_ambiental_alta",
            "severidad": sev_hr,
            "mensaje": "Humedad ambiental elevada; riesgo de reabsorción y moho.",
            "variable": "humedad_ambiental",
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

    sev_estancado = clasificar_secado_estancado(delta_humedad_grano_24h, features.get("humedad_grano"))
    if sev_estancado != "normal":
        alertas.append({
            "tipo": "secado_estancado",
            "severidad": sev_estancado,
            "mensaje": "La humedad del grano no ha bajado en las últimas 24h.",
            "variable": "humedad_grano",
        })

    sev_moho = clasificar_riesgo_moho(features.get("humedad_ambiental"), features.get("temperatura_grano"))
    if sev_moho != "normal" and sev_hr == "normal":
        alertas.append({
            "tipo": "riesgo_moho",
            "severidad": sev_moho,
            "mensaje": "Combinación de humedad y temperatura favorable a formación de moho.",
            "variable": "humedad_ambiental",
        })

    if features.get("luz") is not None and features.get("luz") < LUZ_INSUFICIENTE_LUX and sev_estancado != "normal":
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
    "humedad_ambiental_alta": "Voltea el café con mayor frecuencia y mejora la ventilación de la zona de secado para evitar reabsorción de humedad.",
    "riesgo_moho": "Aumenta la frecuencia de volteo y ventilación: la combinación de humedad y temperatura actual favorece la formación de moho.",
    "lluvia_detectada": "Prioridad máxima: cubre el lote con plástico o lona de inmediato para protegerlo de la lluvia.",
    "secado_estancado": "Revisa que el grosor de la capa esté entre 2 y 10 cm y aumenta la frecuencia de volteo; la humedad del grano no está bajando.",
    "fluctuacion_termica": "Se detectó un cambio brusco de temperatura ('montaña rusa climática'); aumenta la frecuencia de monitoreo preventivo.",
    "radiacion_insuficiente": "Poca luz solar sostenida con secado estancado: considera mover el lote a una zona con más exposición solar o usar secado asistido.",
    "valor_imposible": "Revisa la conexión y calibración del sensor; la última lectura está fuera de rango físico posible.",
    "normal": "El lote se encuentra dentro de los parámetros esperados para su tipo de proceso.",
}


def recomendacion_para(tipo_alerta: str) -> str:
    return RECOMENDACIONES.get(tipo_alerta, RECOMENDACIONES["normal"])


# Severidad "por defecto" asociada a cada tipo de anomalía, usada cuando el
# clasificador RandomForest predice un tipo que la evaluación instantánea de
# reglas no marcó (p.ej. patrones que el modelo generalizó a partir del entrenamiento).
TIPO_SEVERIDAD_DEFAULT: Dict[str, str] = {
    "normal": "normal",
    "temperatura_alta": "riesgo",
    "humedad_ambiental_alta": "riesgo",
    "riesgo_moho": "riesgo",
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
