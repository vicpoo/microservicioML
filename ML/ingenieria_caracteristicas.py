"""
ML/ingenieria_caracteristicas.py

Paso 5 del pipeline de ML — Ingeniería de características.

Entrada:  data/processed/lecturas_reales_limpias.csv   (salida del paso 4, ML/04_limpieza_datos.ipynb)
Salida:   data/processed/lecturas_reales_features.csv  (lista para el paso 6: train/test split)

Este módulo NO vuelve a limpiar datos (eso ya lo hizo el paso 4) ni vuelve a calcular las
etiquetas de reglas (_es_anomalia/_severidad/_tipo_anomalia, ya calculadas en el paso 2 por
scripts/recolectar_datos_reales.py usando app/services/rules.py). Su única responsabilidad es
agregar columnas DERIVADAS que no existen todavía, útiles para los modelos supervisados
(rf_tiempo_restante, rf_calidad) y para el no supervisado (isolation_forest):

  - Estadísticos de ventana móvil (media/desviación de temperatura y luz en la última hora):
    capturan tendencia local, no solo el valor instantáneo.
  - Velocidad de cambio de temperatura_grano (°C/hora): distingue un cambio gradual de un
    salto brusco entre dos lecturas consecutivas.
  - Progreso relativo del proceso (horas_transcurridas / duración típica del tipo de proceso):
    normaliza "qué tan avanzado va" un lote sin importar si es lavado/honey/natural.
  - Codificación cíclica de la hora del día (seno/coseno): para que el modelo entienda que
    las 23:00 y las 00:00 están "cerca", cosa que un entero de hora normal no captura.
  - Señales acumuladas de lluvia (horas desde la última lluvia sostenida, número de eventos
    de lluvia sostenida en las últimas 24h) y de luz (luz acumulada en las últimas 6h, proxy
    de energía solar recibida recientemente).

Se agrupa siempre por id_lote antes de calcular cualquier ventana (rolling/diff), para no
mezclar la serie de tiempo de un lote con la de otro.

Pensado para reutilizarse igual que app/services/rules.py: tanto desde el notebook de este
mismo folder (05_ingenieria_caracteristicas.ipynb, para explorar/graficar) como, más adelante,
desde scripts/train_models.py (para que el dataset de entrenamiento use exactamente las mismas
fórmulas que cualquier análisis exploratorio).
"""
import os

import numpy as np
import pandas as pd

# Duración típica de cada proceso (Cuadro 1 del documento de dominio; mismos valores que
# app/services/rules.py::DURACION_HORAS, aquí ya promediados a un solo número por proceso).
DURACION_HORAS_PROMEDIO = {
    "lavado": (6 * 24 + 9 * 24) / 2,
    "honey": (8 * 24 + 23 * 24) / 2,
    "natural": (10 * 24 + 28 * 24) / 2,
}

VENTANA_ROLLING_TEMP = "60min"
VENTANA_LUZ_ACUMULADA = "6h"
VENTANA_LLUVIA_EVENTOS = "24h"

COLUMNAS_NUEVAS = [
    "temp_grano_media_1h", "temp_grano_std_1h",
    "temp_ambiental_media_1h",
    "velocidad_cambio_temp_grano_c_h",
    "progreso_proceso",
    "hora_sin", "hora_cos",
    "horas_desde_ultima_lluvia",
    "lluvia_eventos_24h",
    "luz_acumulada_6h",
]


def _requiere_columnas(df: pd.DataFrame, columnas) -> None:
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas para ingeniería de características: {faltantes}")


def _rolling_por_lote(df: pd.DataFrame, columna: str, ventana: str, como: str) -> pd.Series:
    """Aplica una función rolling (mean/std/sum) sobre `columna`, agrupando por id_lote y
    ordenando por timestamp dentro de cada grupo, devolviendo una Serie alineada al índice
    original de df (mismo patrón usado en ML/04_limpieza_datos.ipynb para lluvia_sostenida)."""
    partes = []
    for _, grupo in df.groupby("id_lote"):
        g = grupo.sort_values("timestamp").set_index("timestamp")
        rolling = getattr(g[columna].rolling(ventana, min_periods=1), como)()
        partes.append(pd.Series(rolling.values, index=grupo.sort_values("timestamp").index))
    return pd.concat(partes).sort_index()


def _velocidad_cambio_por_lote(df: pd.DataFrame) -> pd.Series:
    """°C de cambio en temperatura_grano por hora transcurrida, contra la lectura anterior
    del mismo lote. None en la primera lectura de cada lote (no hay 'anterior')."""
    partes = []
    for _, grupo in df.groupby("id_lote"):
        g = grupo.sort_values("timestamp")
        delta_temp = g["temperatura_grano"].diff()
        delta_horas = g["timestamp"].diff().dt.total_seconds() / 3600.0
        velocidad = delta_temp / delta_horas.replace(0, np.nan)
        partes.append(pd.Series(velocidad.values, index=g.index))
    return pd.concat(partes).sort_index()


def _horas_desde_ultima_lluvia_por_lote(df: pd.DataFrame, columna_lluvia: str) -> pd.Series:
    """Horas desde la última vez que columna_lluvia fue True en ese mismo lote. Antes de la
    primera lluvia registrada del lote, se deja NaN (no 0) para no fingir que acaba de llover."""
    partes = []
    for _, grupo in df.groupby("id_lote"):
        g = grupo.sort_values("timestamp")
        ultima_lluvia_ts = g["timestamp"].where(g[columna_lluvia].fillna(False)).ffill()
        horas = (g["timestamp"] - ultima_lluvia_ts).dt.total_seconds() / 3600.0
        partes.append(pd.Series(horas.values, index=g.index))
    return pd.concat(partes).sort_index()


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe el dataset limpio (paso 4) y devuelve una COPIA con las columnas nuevas de
    COLUMNAS_NUEVAS agregadas. No modifica ni elimina ninguna columna existente."""
    _requiere_columnas(df, [
        "id_lote", "timestamp", "tipo_proceso", "horas_transcurridas",
        "temperatura_grano", "temperatura_ambiental", "luz",
    ])

    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["id_lote", "timestamp"]).reset_index(drop=True)

    # Columna de lluvia "limpia" a usar para features acumuladas: preferir lluvia_sostenida
    # (paso 4) si existe; si no, caer de vuelta a lluvia_detectada cruda con una advertencia.
    if "lluvia_sostenida" in df.columns:
        columna_lluvia = "lluvia_sostenida"
    else:
        print("[ingenieria_caracteristicas] AVISO: no existe 'lluvia_sostenida' (¿se saltó el "
              "paso 4?); usando 'lluvia_detectada' cruda para las features de lluvia acumulada, "
              "lo que puede sobreestimar eventos por el parpadeo conocido del sensor FC-37.")
        columna_lluvia = "lluvia_detectada"

    df["temp_grano_media_1h"] = _rolling_por_lote(df, "temperatura_grano", VENTANA_ROLLING_TEMP, "mean")
    df["temp_grano_std_1h"] = _rolling_por_lote(df, "temperatura_grano", VENTANA_ROLLING_TEMP, "std")
    df["temp_ambiental_media_1h"] = _rolling_por_lote(df, "temperatura_ambiental", VENTANA_ROLLING_TEMP, "mean")

    df["velocidad_cambio_temp_grano_c_h"] = _velocidad_cambio_por_lote(df)

    duracion = df["tipo_proceso"].str.lower().map(DURACION_HORAS_PROMEDIO).fillna(DURACION_HORAS_PROMEDIO["lavado"])
    df["progreso_proceso"] = (df["horas_transcurridas"] / duracion).clip(lower=0, upper=2)

    hora_decimal = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hora_sin"] = np.sin(2 * np.pi * hora_decimal / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora_decimal / 24)

    df["horas_desde_ultima_lluvia"] = _horas_desde_ultima_lluvia_por_lote(df, columna_lluvia)

    lluvia_int = df[columna_lluvia].fillna(False).astype(int)
    df_temp = df.copy()
    df_temp["_lluvia_int"] = lluvia_int
    df["lluvia_eventos_24h"] = _rolling_por_lote(df_temp, "_lluvia_int", VENTANA_LLUVIA_EVENTOS, "sum")

    df["luz_acumulada_6h"] = _rolling_por_lote(df, "luz", VENTANA_LUZ_ACUMULADA, "sum")

    return df


def main():
    here = os.path.dirname(__file__)
    ruta_entrada = os.path.join(here, "..", "data", "processed", "lecturas_reales_limpias.csv")
    ruta_salida = os.path.join(here, "..", "data", "processed", "lecturas_reales_features.csv")

    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(
            f"No existe {ruta_entrada}. Corre primero ML/04_limpieza_datos.ipynb (paso 4)."
        )

    df = pd.read_csv(ruta_entrada, parse_dates=["timestamp"])

    # Paso 5 opera solo sobre lotes vigentes: un lote 'cancelado' (p.ej. un sensor todavía sin
    # vincular a un usuario/lote real) no debería usarse para construir features de entrenamiento,
    # aunque el paso 4 lo haya conservado por volumen de lecturas.
    antes = len(df)
    if "estado_lote" in df.columns:
        df = df[df["estado_lote"] != "cancelado"].copy()
    print(f"Filas tras excluir lotes cancelados: {len(df):,} (de {antes:,})")

    df_features = construir_features(df)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_features.to_csv(ruta_salida, index=False)
    print(f"Guardado: {ruta_salida}  ({len(df_features):,} filas, {len(df_features.columns)} columnas)")
    print(f"Columnas nuevas agregadas: {COLUMNAS_NUEVAS}")


if __name__ == "__main__":
    main()
