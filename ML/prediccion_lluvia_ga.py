"""
ML/prediccion_lluvia_ga.py

Paso 7 (selección del modelo) — pieza de "predicción de lluvia" del pipeline, resuelta con un
Algoritmo Genético (AG), a diferencia de las demás salidas (Random Forest / IsolationForest).

Por qué un AG y no un clasificador estándar aquí:
  - El resto del sistema (app/services/rules.py) ya es 100% reglas de umbral interpretables
    (Cuadros 9/10 del documento de dominio). Un AG permite seguir en ese mismo espíritu:
    en vez de que un experto humano adivine a mano los pesos/umbrales de una regla de "riesgo de
    lluvia próxima" (usando presión, luz, racha reciente de lluvia), el AG los *evoluciona*
    optimizando directamente F1 sobre datos reales -- pero el resultado sigue siendo una regla
    lineal con pesos explícitos por variable, auditable como cualquier otra regla del sistema
    (no es una caja negra tipo Random Forest/red neuronal).
  - Es más apropiado que un clasificador de caja negra para una señal que, con los sensores
    disponibles (BMP280 + BH1750, sin estación meteorológica externa), tiene pocas variables
    realmente predictivas -- un espacio de búsqueda pequeño e interpretable es ideal para un AG.

Qué predice exactamente: no "está lloviendo ahora" (eso ya lo hace directo el sensor FC-37,
`lluvia_detectada`/`lluvia_sostenida`, ver paso 4). Predice **riesgo de que llueva en las
próximas H horas**, a partir de condiciones ACTUALES (presión, luz, cuánto ha llovido
recientemente) -- una predicción real hacia adelante, no una simple relectura del sensor.

Individuo del AG: vector [w_presion, w_luz, w_eventos_24h, w_horas_desde_ultima_lluvia, bias].
Score = w · x_normalizado + bias; predicción = 1 (riesgo de lluvia) si score > 0.
Fitness = F1-score de esa predicción contra la etiqueta real "llovió en las próximas H horas".
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

FEATURES_GA = ["presion_hpa", "luz", "lluvia_eventos_24h", "horas_desde_ultima_lluvia"]
HORAS_ANTICIPACION_DEFAULT = 3


# --- 1. Etiqueta objetivo: ¿llovió (lluvia_sostenida) en las próximas H horas? ---------------

def etiquetar_lluvia_proxima(
    df: pd.DataFrame,
    horas: int = HORAS_ANTICIPACION_DEFAULT,
    id_col: str = "id_lote",
    timestamp_col: str = "timestamp",
    col_lluvia: str = "lluvia_sostenida",
) -> pd.Series:
    """Para cada lectura, revisa si `col_lluvia` es True en algún punto dentro de las
    siguientes `horas` horas (mismo lote). Si no queda suficiente "futuro" dentro del propio
    lote para saberlo con certeza (lecturas de las últimas `horas` horas de cada lote), la
    etiqueta queda NaN -- esas filas se descartan al entrenar/evaluar, no se adivinan.

    Implementación O(n log n) por lote: usa búsqueda binaria (searchsorted) sobre timestamps
    ordenados + suma acumulada de `col_lluvia` para contar eventos futuros sin un loop O(n²).
    """
    partes = []
    ventana_ns = np.timedelta64(int(horas * 3600), "s")
    for _, grupo in df.groupby(id_col):
        g = grupo.sort_values(timestamp_col).reset_index(drop=True)
        tiempos = g[timestamp_col].values.astype("datetime64[ns]")
        lluvia = g[col_lluvia].fillna(False).values.astype(bool)
        n = len(g)

        limite_ventana = tiempos + ventana_ns
        # fin_idx[i] = primer índice j tal que tiempos[j] > limite_ventana[i] (búsqueda binaria)
        fin_idx = np.searchsorted(tiempos, limite_ventana, side="right")
        cum_lluvia = np.concatenate([[0], np.cumsum(lluvia.astype(int))])

        etiquetas = np.full(n, np.nan)
        ultimo_tiempo = tiempos[-1] if n else None
        for i in range(n):
            if limite_ventana[i] > ultimo_tiempo:
                continue  # no hay suficiente futuro en este lote para confirmar/descartar
            j = fin_idx[i]
            suma_futuro = cum_lluvia[j] - cum_lluvia[i + 1] if j > i + 1 else 0
            etiquetas[i] = 1.0 if suma_futuro > 0 else 0.0

        partes.append(pd.Series(etiquetas, index=grupo.index))
    return pd.concat(partes).sort_index()


# --- 2. Escalado simple (min-max, ajustado solo con train) ----------------------------------

@dataclass
class EscaladorMinMax:
    minimos: Optional[np.ndarray] = None
    maximos: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "EscaladorMinMax":
        self.minimos = np.nanmin(X, axis=0)
        self.maximos = np.nanmax(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rango = np.where((self.maximos - self.minimos) == 0, 1.0, self.maximos - self.minimos)
        return (X - self.minimos) / rango


def preparar_X_y(df: pd.DataFrame, etiqueta_col: str = "_lluvia_proxima") -> Tuple[np.ndarray, np.ndarray]:
    datos = df.dropna(subset=FEATURES_GA + [etiqueta_col])
    X = datos[FEATURES_GA].to_numpy(dtype=float)
    y = datos[etiqueta_col].to_numpy(dtype=float)
    return X, y


# --- 3. Algoritmo genético --------------------------------------------------------------------

def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def predecir(individuo: np.ndarray, X: np.ndarray) -> np.ndarray:
    """individuo = [w_1..w_k, bias]. score = X·w + bias; predice 1 si score > 0."""
    pesos, bias = individuo[:-1], individuo[-1]
    score = X @ pesos + bias
    return (score > 0).astype(int)


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 promediado entre ambas clases (0 y 1), no solo la clase 1 como `_f1`. Penaliza un
    individuo que solo acierta prediciendo siempre la clase mayoritaria -- ver paso 9
    (09_evaluacion.ipynb): con fitness=F1 puro, el AG convergió a "predecir lluvia siempre" en un
    tramo de test desbalanceado, porque esa estrategia ya maximiza F1(clase 1) sin penalización
    por ignorar la clase 0 por completo."""
    f1_clase1 = _f1(y_true, y_pred)
    f1_clase0 = _f1(1 - y_true, 1 - y_pred)
    return (f1_clase1 + f1_clase0) / 2.0


_METRICAS_FITNESS = {"f1": _f1, "f1_macro": _f1_macro}


def _fitness(individuo: np.ndarray, X: np.ndarray, y: np.ndarray, metrica: str = "f1") -> float:
    funcion = _METRICAS_FITNESS[metrica]
    return funcion(y, predecir(individuo, X))


@dataclass
class ResultadoAG:
    mejor_individuo: np.ndarray
    historial_fitness: List[float] = field(default_factory=list)
    escalador: Optional[EscaladorMinMax] = None


def evolucionar(
    X: np.ndarray,
    y: np.ndarray,
    generaciones: int = 40,
    tam_poblacion: int = 60,
    prob_mutacion: float = 0.2,
    escala_mutacion: float = 0.3,
    tam_torneo: int = 3,
    semilla: int = 42,
    metrica_fitness: str = "f1",
) -> ResultadoAG:
    if metrica_fitness not in _METRICAS_FITNESS:
        raise ValueError(f"metrica_fitness debe ser una de {list(_METRICAS_FITNESS)}")
    rng = np.random.default_rng(semilla)
    n_genes = X.shape[1] + 1  # pesos + bias

    poblacion = rng.uniform(-1, 1, size=(tam_poblacion, n_genes))
    historial = []

    def evaluar_poblacion(pobl):
        return np.array([_fitness(ind, X, y, metrica=metrica_fitness) for ind in pobl])

    fitness_actual = evaluar_poblacion(poblacion)

    for _ in range(generaciones):
        nueva_poblacion = []

        # elitismo: el mejor individuo pasa intacto
        idx_mejor = int(np.argmax(fitness_actual))
        nueva_poblacion.append(poblacion[idx_mejor].copy())

        while len(nueva_poblacion) < tam_poblacion:
            # selección por torneo
            def torneo():
                idxs = rng.integers(0, tam_poblacion, size=tam_torneo)
                return poblacion[idxs[np.argmax(fitness_actual[idxs])]]

            padre1, padre2 = torneo(), torneo()

            # cruza aritmética (blend uniforme por gen)
            mascara = rng.random(n_genes) < 0.5
            hijo = np.where(mascara, padre1, padre2)

            # mutación gaussiana
            if rng.random() < prob_mutacion:
                hijo = hijo + rng.normal(0, escala_mutacion, size=n_genes)

            nueva_poblacion.append(hijo)

        poblacion = np.array(nueva_poblacion)
        fitness_actual = evaluar_poblacion(poblacion)
        historial.append(float(np.max(fitness_actual)))

    idx_mejor = int(np.argmax(fitness_actual))
    return ResultadoAG(mejor_individuo=poblacion[idx_mejor], historial_fitness=historial)


# --- 4. Función de conveniencia: entrenar + evaluar de punta a punta -------------------------

def etiquetar_particiones(particiones: dict, horas: int = HORAS_ANTICIPACION_DEFAULT) -> dict:
    """Generalización de `etiquetar_train_test` a N particiones temporales consecutivas del MISMO
    sensor/línea de tiempo (p. ej. train_fit/train_val/test al afinar hiperparámetros -- ver
    ML/ajuste_hiperparametros.py). Calcula `_lluvia_proxima` sobre la unión de TODAS las
    particiones para que ninguna quede con NaN solo por caer cerca del borde de una partición
    (ver docstring de `etiquetar_train_test` para el porqué de fondo); cada partición sigue
    ajustándose/puntuándose SOLO con sus propias filas, la unión es únicamente para construir la
    etiqueta histórica correctamente."""
    marca_col = "_particion_temporal__"
    piezas = [df.assign(**{marca_col: nombre}) for nombre, df in particiones.items()]
    completo = pd.concat(piezas, ignore_index=True).sort_values(["id_lote", "timestamp"]).reset_index(drop=True)
    completo["_lluvia_proxima"] = etiquetar_lluvia_proxima(completo, horas=horas)

    return {
        nombre: completo[completo[marca_col] == nombre].drop(columns=[marca_col])
        for nombre in particiones
    }


def etiquetar_train_test(
    df_train: pd.DataFrame, df_test: pd.DataFrame, horas: int = HORAS_ANTICIPACION_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula `_lluvia_proxima` sobre la línea de tiempo COMPLETA (train + test unidos por
    lote), no por separado en cada partición.

    Por qué: si se etiqueta train y test cada uno por su cuenta, cualquier lectura dentro de las
    últimas `horas` horas de TRAIN queda con NaN aunque el desenlace real sí exista (justo al
    inicio de test) -- se pierde información real por un artefacto de dónde cae el corte del
    paso 6, no por falta de datos genuina. Unir ambas particiones (que son, al final, el mismo
    sensor/línea de tiempo continua) para construir la etiqueta es válido: no se usa ninguna
    FEATURE del futuro como insumo del modelo, solo se usa el desenlace real ya ocurrido para
    fijar la etiqueta de filas históricas -- exactamente como cualquier etiqueta de pronóstico de
    series de tiempo requiere. Solo las lecturas dentro de las últimas `horas` horas del dataset
    COMPLETO (el final real de test) se quedan sin etiqueta, por falta de datos genuina."""
    resultado = etiquetar_particiones({"train": df_train, "test": df_test}, horas=horas)
    return resultado["train"], resultado["test"]


def entrenar_y_evaluar(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    horas: int = HORAS_ANTICIPACION_DEFAULT,
    generaciones: int = 40,
    tam_poblacion: int = 60,
    semilla: int = 42,
    metrica_fitness: str = "f1",
) -> dict:
    df_train, df_test = etiquetar_train_test(df_train, df_test, horas=horas)

    X_train_crudo, y_train = preparar_X_y(df_train)
    X_test_crudo, y_test = preparar_X_y(df_test)

    escalador = EscaladorMinMax().fit(X_train_crudo)
    X_train = escalador.transform(X_train_crudo)
    X_test = escalador.transform(X_test_crudo)

    resultado = evolucionar(
        X_train, y_train, generaciones=generaciones, tam_poblacion=tam_poblacion,
        semilla=semilla, metrica_fitness=metrica_fitness,
    )

    y_pred_train = predecir(resultado.mejor_individuo, X_train)
    y_pred_test = predecir(resultado.mejor_individuo, X_test)

    return {
        "mejor_individuo": dict(zip(FEATURES_GA + ["bias"], resultado.mejor_individuo.tolist())),
        "historial_fitness": resultado.historial_fitness,
        "metrica_fitness": metrica_fitness,
        "f1_train": _f1(y_train, y_pred_train),
        "f1_test": _f1(y_test, y_pred_test),
        "f1_macro_train": _f1_macro(y_train, y_pred_train),
        "f1_macro_test": _f1_macro(y_test, y_pred_test),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "positivos_train": int(y_train.sum()),
        "positivos_test": int(y_test.sum()),
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "escalador": escalador,
    }
