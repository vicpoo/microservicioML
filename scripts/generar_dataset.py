#scripts/generar_dataset.py
"""
Genera un dataset sintético de secado de café para entrenar los 3 modelos del pipeline:
  1. Detección de anomalías (IsolationForest, no supervisado)
  2. Clasificación de tipo/severidad de anomalía (RandomForestClassifier, supervisado)
  3. Predicción de horas restantes de secado (RandomForestRegressor, supervisado)
  4. Predicción de calidad final estimada (RandomForestClassifier, supervisado)

Simula lotes completos (no lecturas sueltas) para que las etiquetas de "horas restantes"
y "calidad final" tengan sentido temporal/agregado por lote. Usa las MISMAS reglas de
dominio (app/services/rules.py) que usará el servicio en producción, así el modelo
aprende a generalizar esas reglas y no queda desalineado con ellas.

Sensores simulados: BME280 (temperatura_ambiental, humedad_ambiental),
DS18B20 (temperatura_grano), BH1750 (luz), FC-37 (lluvia),
sensor capacitivo de humedad (humedad_grano). Sin anemómetro: no se genera "viento".

Salida: data/raw/lecturas_ml_training.csv
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.rules import DURACION_HORAS, evaluar_lectura  # noqa: E402

RNG = np.random.default_rng(42)
PROCESOS = ["lavado", "honey", "natural"]
LOTES_POR_PROCESO = 24
PASO_HORAS = 2

TEMP_IDEAL_MEDIO = {"lavado": 27.0, "honey": 28.0, "natural": 30.0}


def simular_lote(id_lote: int, tipo_proceso: str):
    lo, hi = DURACION_HORAS[tipo_proceso]
    total_horas = int(RNG.uniform(lo, hi))
    descuido = RNG.beta(2, 5)  # 0 = productor muy cuidadoso, 1 = muy descuidado
    humedad_grano_inicial = RNG.uniform(45, 57)
    humedad_grano_final = RNG.uniform(10, 12)
    temp_medio = TEMP_IDEAL_MEDIO[tipo_proceso] + descuido * RNG.uniform(2, 8)
    cloud_factor_lote = RNG.uniform(0.6, 1.0)

    filas = []
    humedad_grano_prev = humedad_grano_inicial
    temp_grano_prev = temp_medio
    racha_lluvia_restante = 0
    racha_lluvia_intensidad = 0.0

    for t in range(0, total_horas + 1, PASO_HORAS):
        hora_dia = t % 24
        progreso = t / max(total_horas, 1)

        # --- Luz (BH1750): curva diurna, 0 de noche, pico ~12-14h ---
        luz_dia = max(0.0, math.sin(math.pi * (hora_dia - 6) / 12))
        pico_luz = RNG.uniform(60000, 100000) * cloud_factor_lote
        luz = max(0.0, luz_dia * pico_luz + RNG.normal(0, 1500))

        # --- Lluvia (FC-37): rachas aleatorias, más probables si el productor es descuidado ---
        if racha_lluvia_restante > 0:
            lluvia = max(0.0, min(1.0, racha_lluvia_intensidad + RNG.normal(0, 0.05)))
            racha_lluvia_restante -= 1
        else:
            prob_lluvia = 0.008 + 0.006 * descuido
            if RNG.uniform() < prob_lluvia:
                racha_lluvia_restante = int(RNG.integers(1, 4))
                racha_lluvia_intensidad = RNG.uniform(0.3, 1.0)
                lluvia = racha_lluvia_intensidad
            else:
                lluvia = 0.0

        # --- Temperatura ambiental (BME280): diurna + ruido ---
        amplitud_temp = RNG.uniform(4, 7)
        temperatura_ambiental = (
            temp_medio - 3 + amplitud_temp * math.sin(math.pi * (hora_dia - 8) / 12) + RNG.normal(0, 0.8)
        )
        if lluvia > 0:
            temperatura_ambiental -= RNG.uniform(2, 5)

        # --- Humedad ambiental (BME280): inversa a temp + lluvia sube mucho ---
        humedad_ambiental = 48 - 0.9 * (temperatura_ambiental - temp_medio) + RNG.normal(0, 3)
        humedad_ambiental += descuido * 6  # productores descuidados secan en peores condiciones
        if lluvia > 0:
            humedad_ambiental += 25 + lluvia * 10
        humedad_ambiental = float(np.clip(humedad_ambiental, 15, 99))

        # --- Temperatura de grano (DS18B20): sigue a la ambiental + calor de exposición solar ---
        sobreexposicion = max(0.0, (luz / 100000) - 0.4) * (5 + descuido * 6)
        temperatura_grano = temperatura_ambiental + 2 + sobreexposicion + RNG.normal(0, 0.6)

        # --- Humedad de grano: decrece hacia el objetivo, se estanca con lluvia/HR alta ---
        avance_ideal = humedad_grano_inicial - (humedad_grano_inicial - humedad_grano_final) * min(progreso * 1.05, 1.0)
        factor_estancamiento = 1.0
        if humedad_ambiental > 80 or lluvia > 0:
            factor_estancamiento = RNG.uniform(0.1, 0.5)
        humedad_grano = humedad_grano_prev - (humedad_grano_prev - avance_ideal) * 0.15 * factor_estancamiento
        humedad_grano = float(np.clip(humedad_grano + RNG.normal(0, 0.3), humedad_grano_final - 1, 100))

        # --- Fallas de sensor ocasionales: valor imposible / lectura repetida ---
        valor_imposible = RNG.uniform() < 0.004
        if valor_imposible:
            if RNG.uniform() < 0.5:
                temperatura_grano = RNG.uniform(90, 120)
            else:
                humedad_ambiental = RNG.uniform(-5, 3)
        lectura_estancada = RNG.uniform() < 0.006
        if lectura_estancada and filas:
            temperatura_grano = temp_grano_prev
            humedad_grano = humedad_grano_prev

        # ~3% de lecturas nulas (falla de MQTT/sensor), se completan luego en limpieza
        nulo = RNG.uniform() < 0.03

        delta_temp_reciente = abs(temperatura_grano - temp_grano_prev) if filas else 0.0
        delta_humedad_grano_24h = None
        if t >= 24:
            idx_24h = max(0, len(filas) - int(24 / PASO_HORAS))
            delta_humedad_grano_24h = filas[idx_24h]["humedad_grano"] - humedad_grano

        features = {
            "temperatura_grano": round(temperatura_grano, 2),
            "temperatura_ambiental": round(temperatura_ambiental, 2),
            "humedad_ambiental": round(humedad_ambiental, 2),
            "humedad_grano": round(humedad_grano, 2),
            "lluvia": round(lluvia, 3),
            "luz": round(luz, 1),
        }
        evaluacion = evaluar_lectura(
            tipo_proceso, features, delta_temp_reciente=delta_temp_reciente, delta_humedad_grano_24h=delta_humedad_grano_24h
        )

        fila = {
            "id_lote": id_lote,
            "tipo_proceso": tipo_proceso,
            "horas_transcurridas": t,
            "horas_restantes": max(total_horas - t, 0),
            **features,
            "_es_anomalia": evaluacion["es_anomalia"],
            "_severidad": evaluacion["severidad"],
            "_tipo_anomalia": evaluacion["tipo_principal"],
        }
        if nulo:
            for campo in ("temperatura_grano", "humedad_ambiental"):
                fila[campo] = np.nan
        filas.append(fila)

        humedad_grano_prev = humedad_grano
        temp_grano_prev = temperatura_grano

    return pd.DataFrame(filas)


def asignar_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna calidad_final por lote a partir de un score de riesgo acumulado
    (Cuadro 10: crítico pesa más que riesgo, que pesa más que advertencia),
    usando cuantiles sobre la población simulada para obtener una distribución
    de calidad realista (no todo-o-nada)."""
    resumen = (
        df.groupby("id_lote")["_severidad"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    for col in ("critico", "riesgo", "advertencia", "normal"):
        if col not in resumen:
            resumen[col] = 0.0
    resumen["score_riesgo"] = resumen["critico"] * 4 + resumen["riesgo"] * 2 + resumen["advertencia"] * 0.5

    etiquetas = ["excelente", "buena", "regular", "baja"]
    resumen["_calidad_final_lote"] = pd.qcut(
        resumen["score_riesgo"].rank(method="first"), q=[0, 0.25, 0.60, 0.85, 1.0], labels=etiquetas
    )
    return df.merge(resumen["_calidad_final_lote"], on="id_lote", how="left")


def main():
    lotes = []
    id_lote = 1
    for proceso in PROCESOS:
        for _ in range(LOTES_POR_PROCESO):
            lotes.append(simular_lote(id_lote, proceso))
            id_lote += 1
    df = pd.concat(lotes, ignore_index=True)
    df = asignar_calidad(df)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lecturas_ml_training.csv")
    df.to_csv(out_path, index=False)

    print(f"Lecturas totales: {len(df):,}")
    print(f"Lotes simulados: {df['id_lote'].nunique()}")
    print(f"Anomalias: {df['_es_anomalia'].sum():,} ({df['_es_anomalia'].mean()*100:.1f}%)")
    print("\nDistribucion severidad:")
    print(df["_severidad"].value_counts().to_string())
    print("\nDistribucion tipo_anomalia:")
    print(df["_tipo_anomalia"].value_counts().to_string())
    print("\nCalidad final por lote:")
    print(df.drop_duplicates("id_lote")["_calidad_final_lote"].value_counts().to_string())
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
