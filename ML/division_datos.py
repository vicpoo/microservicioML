"""
ML/division_datos.py

Paso 6 del pipeline de ML — División de datos (train/test).

Entrada:  data/processed/lecturas_reales_features.csv  (salida del paso 5)
Salida:   data/processed/train.csv, data/processed/test.csv

Por qué esto NO puede ser un train_test_split aleatorio de filas:
  1. Fuga de datos entre lecturas del mismo lote: dos lecturas consecutivas del mismo sensor
     están fuertemente correlacionadas (además, el paso 5 ya agregó columnas de ventana móvil
     como temp_grano_media_1h, que literalmente promedian lecturas vecinas). Si el split es
     aleatorio a nivel de fila, el modelo ve en entrenamiento lecturas a segundos de distancia de
     otras que están en "test" -> métricas de evaluación artificialmente optimistas.
  2. Fuga de datos entre lotes: si un mismo lote aporta filas tanto a train como a test, el
     modelo puede memorizar patrones específicos de ESE lote (su sensor, su microclima) en vez de
     aprender algo que generalice a un lote nuevo. Por eso el estándar correcto es dividir por
     GRUPO (id_lote completo va a train o a test, nunca partido).

El problema real de este proyecto en esta etapa (ver definicion_problema_kajve.md, Sección 6):
  hoy existe **un solo lote real con volumen utilizable** (id_lote=12). Con un solo grupo, una
  división por lote (GroupShuffleSplit) es matemáticamente imposible: no se puede partir 1 grupo
  en 2 grupos no vacíos. `entrenar_clasificador_tipo/tiempo/calidad` en scripts/train_models.py ya
  detectan esto y se saltan el entrenamiento -- pero eso significa que, tal cual está hoy, NINGÚN
  modelo supervisado se valida todavía con datos reales.

Este módulo resuelve la división de datos en dos modos, eligiendo automáticamente según cuántos
lotes reales hay:
  - `metodo="group_shuffle_split_por_lote"`: cuando hay >= min_lotes_para_grupo lotes (el ideal,
    listo para cuando el piloto crezca). Ningún lote queda partido entre train y test.
  - `metodo="temporal_por_lote"`: fallback de HOY. Dentro de cada lote, ordena por tiempo y usa
    el tramo cronológico más reciente como test (nunca lecturas intercaladas al azar). Esto
    permite evaluar honestamente "qué tan bien generaliza el modelo a las horas más recientes de
    secado que no vio en entrenamiento", que es la única validación posible con un solo lote real,
    sin caer en el problema de fuga por ventanas móviles del punto 1.
"""
import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TEST_SIZE_DEFAULT = 0.25
MIN_LOTES_PARA_GRUPO = 2
RANDOM_STATE = 42


def _split_temporal_por_lote(df: pd.DataFrame, id_col: str, timestamp_col: str, test_size: float):
    """Para cada lote, ordena por tiempo y separa el tramo final (cronológico) como test.
    Devuelve (train_idx, test_idx) como arrays de posiciones (no de índice de pandas)."""
    train_pos, test_pos = [], []
    df_reset = df.reset_index(drop=True)
    for _, grupo in df_reset.groupby(id_col):
        g = grupo.sort_values(timestamp_col)
        n_test = max(1, int(round(len(g) * test_size)))
        train_pos.extend(g.index[:-n_test].tolist())
        test_pos.extend(g.index[-n_test:].tolist())
    return sorted(train_pos), sorted(test_pos)


def dividir_datos(
    df: pd.DataFrame,
    id_col: str = "id_lote",
    timestamp_col: str = "timestamp",
    test_size: float = TEST_SIZE_DEFAULT,
    min_lotes_para_grupo: int = MIN_LOTES_PARA_GRUPO,
    random_state: int = RANDOM_STATE,
):
    """Divide df en train/test eligiendo automáticamente el método según cuántos lotes hay.

    Devuelve: (df_train, df_test, info) donde info es un dict con al menos:
      {"metodo": ..., "n_lotes": ..., "n_train": ..., "n_test": ...}
    """
    df = df.reset_index(drop=True)
    n_lotes = df[id_col].nunique()

    if n_lotes >= min_lotes_para_grupo:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df[id_col]))
        metodo = "group_shuffle_split_por_lote"
    else:
        train_idx, test_idx = _split_temporal_por_lote(df, id_col, timestamp_col, test_size)
        metodo = "temporal_por_lote"

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    info = {
        "metodo": metodo,
        "n_lotes": int(n_lotes),
        "min_lotes_para_grupo": min_lotes_para_grupo,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "lotes_train": sorted(df_train[id_col].unique().tolist()),
        "lotes_test": sorted(df_test[id_col].unique().tolist()),
    }
    return df_train, df_test, info


def main():
    here = os.path.dirname(__file__)
    ruta_entrada = os.path.join(here, "..", "data", "processed", "lecturas_reales_features.csv")
    ruta_train = os.path.join(here, "..", "data", "processed", "train.csv")
    ruta_test = os.path.join(here, "..", "data", "processed", "test.csv")

    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(
            f"No existe {ruta_entrada}. Corre primero ML/05_ingenieria_caracteristicas.ipynb (paso 5)."
        )

    df = pd.read_csv(ruta_entrada, parse_dates=["timestamp"])
    df_train, df_test, info = dividir_datos(df)

    print(f"Método usado: {info['metodo']}")
    print(f"Lotes totales: {info['n_lotes']}  |  lotes en train: {info['lotes_train']}  |  lotes en test: {info['lotes_test']}")
    print(f"Filas train: {info['n_train']:,}  |  Filas test: {info['n_test']:,}")

    df_train.to_csv(ruta_train, index=False)
    df_test.to_csv(ruta_test, index=False)
    print(f"Guardado: {ruta_train}")
    print(f"Guardado: {ruta_test}")


if __name__ == "__main__":
    main()
