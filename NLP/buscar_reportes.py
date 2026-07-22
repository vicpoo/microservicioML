#NLP/buscar_reportes.py
"""
NLP/buscar_reportes.py

Paso 2 de la opción A: buscador de reportes históricos con BM25 -- el mismo algoritmo que ya
usa `rankear_eventos.py` (Nivel 2 del reporte, resumen extractivo dentro de UN reporte), pero
aplicado como lo que realmente es: un motor de búsqueda, sobre TODO el historial de reportes
ya generados de un usuario (todos sus lotes, no solo uno).

Diferencia con `rankear_eventos.py`:
  - `rankear_eventos.eventos_destacados()` rankea mensajes de alerta DENTRO de un mismo reporte,
    contra un query FIJO de urgencia -- siempre la misma pregunta ("¿qué tan urgente es esto?").
  - `buscar_reportes()` rankea reportes completos (`reportes_lote.reporte_texto`) contra un
    query LIBRE que escribe quien busca (ej. "lluvia crítica", "secado estancado") -- la misma
    idea que un buscador de texto normal, aplicada al historial propio del usuario.

Este módulo, igual que `rankear_eventos.py` y `generar_reporte.py`, NO toca la base de datos --
recibe el corpus ya armado (lista de `(id_reporte, texto)`) y regresa los más relevantes. Quien
arma el corpus (consulta a `reportes_lote` filtrada por dueño) es el endpoint del paso 3
(`app/api/routes/history.py`), igual que `recopilar_datos_reporte.py` es quien arma los datos
para `generar_reporte.py`. Separar "quién junta los datos" de "quién los rankea" es el mismo
principio de una sola responsabilidad por módulo que ya sigue el resto del proyecto.

Tokenizador y stopwords compartidos con `rankear_eventos.py` vía `NLP/texto_utils.py` (paso 1):
mismo criterio de "una sola fuente de verdad" para cómo se leen las palabras en todo el PLN.
"""
from dataclasses import dataclass
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from NLP.texto_utils import tokenizar

# Cuántos resultados regresar como máximo cuando no se pide un número distinto.
TOP_N_RESULTADOS_DEFAULT = 5

# Con menos documentos que esto, el IDF de BM25 es matemáticamente degenerado (ver docstring de
# buscar_reportes): se usa un fallback simple de conteo de tokens en común en vez de BM25.
UMBRAL_CORPUS_CHICO = 3


@dataclass
class ResultadoBusqueda:
    id_reporte: int
    texto: str
    score: float


def buscar_reportes(
    corpus: List[Tuple[int, str]], query: str, top_n: int = TOP_N_RESULTADOS_DEFAULT
) -> List[ResultadoBusqueda]:
    """Recibe el corpus de reportes de UN usuario (ya filtrado por dueño por quien llama --
    este módulo no sabe nada de usuarios ni de BD) como pares `(id_reporte, texto)`, y el texto
    de búsqueda libre. Regresa los `top_n` reportes más relevantes según BM25, de más a menos
    relevante, descartando los que no tienen ninguna relación real con el query (score <= 0 --
    BM25 les da 0 cuando ningún término del query aparece en el documento; devolverlos
    "ordenados" de todas formas sería mostrar resultados que no tienen nada que ver con lo que
    se buscó, peor que no mostrar nada).

    A diferencia de `rankear_eventos.eventos_destacados()`, aquí SÍ se rankea aunque haya pocos
    documentos (2 o 3): a diferencia del resumen extractivo (donde con pocos eventos ya no hace
    falta elegir), en una búsqueda real la relevancia de cada documento contra el query varía
    sin importar cuántos haya, y tampoco se deduplica texto -- dos reportes de lotes distintos
    pueden coincidir en texto exacto (mismo patrón de alertas) y ambos son resultados válidos y
    distintos (fechas/lotes diferentes), no un duplicado a filtrar.

    Nota sobre corpus chicos (usuarios con pocos lotes/reportes todavía): BM25 pondera cada
    término por qué tan RARO es en el corpus (IDF) -- si un término del query aparece en
    exactamente la mitad de los reportes de ese usuario, su IDF matemáticamente da 0 y no aporta
    nada al score (comprobado con un caso real: "lluvia" en 2 de 4 reportes = score 0 en los 4).
    No es un bug: para un buscador, un término que aparece en la mitad de tus documentos no
    discrimina nada entre ellos. Con más reportes acumulados por usuario este efecto se diluye
    solo (ver el mismo caso con 6 reportes en vez de 4, donde sí regresa resultados). Se deja
    documentado para no sorprenderse si una búsqueda con muchos empates exactos de 50% regresa
    vacío para ESE término puntual.

    Caso aparte, y más grave, con corpus de 1 o 2 documentos (el usuario más común: alguien que
    apenas empieza y solo tiene uno o dos reportes generados): ahí el IDF de BM25 es degenerado
    de verdad, no solo en un término -- con un solo documento, la "rareza" de cualquier palabra
    no tiene sentido matemático (aparece en el 100% del corpus por definición) y el algoritmo da
    scores negativos para TODO, sin importar el query (comprobado: corpus de 1 reporte que
    contiene la palabra buscada -> BM25 igual da un score negativo, se filtraría como "sin
    relación" siendo falso). Por eso, con menos de `UMBRAL_CORPUS_CHICO` documentos, esta función
    NO usa BM25: cuenta cuántos tokens del query aparecen literalmente en cada documento y
    ordena por esa cuenta -- simple, pero funciona para el caso más común (usuario nuevo) en vez
    de devolver siempre vacío ahí.
    """
    if not corpus:
        return []

    query_tokens = tokenizar(query)
    if not query_tokens:
        # Query vacío o solo stopwords/símbolos: no hay nada específico que buscar. Devolver
        # "todo el historial sin criterio" sería confuso -- mejor no regresar nada, igual que
        # un buscador real no muestra resultados para una caja de búsqueda vacía.
        return []

    ids, textos = zip(*corpus)
    corpus_tokenizado = [tokenizar(t) for t in textos]

    def _por_coincidencia_de_tokens() -> List[ResultadoBusqueda]:
        """Fallback sin BM25: cuenta cuántos tokens del query aparecen literalmente en cada
        documento. Se usa (a) siempre que el corpus es demasiado chico para que el IDF de BM25
        tenga sentido, y (b) como red de seguridad cuando BM25 SÍ corrió pero no encontró ningún
        score positivo -- puede pasar con corpus más grandes si los términos del query caen
        justo en un empate de frecuencia degenerado (ver docstring), y ahí es mejor mostrar algo
        basado en coincidencia literal que nada, cuando el texto sí contiene la palabra buscada."""
        query_set = set(query_tokens)
        candidatos = [
            ResultadoBusqueda(id_reporte=ids[i], texto=textos[i], score=float(len(query_set & set(tokens))))
            for i, tokens in enumerate(corpus_tokenizado)
            if query_set & set(tokens)
        ]
        candidatos.sort(key=lambda r: r.score, reverse=True)
        return candidatos[:top_n]

    if len(corpus) < UMBRAL_CORPUS_CHICO:
        return _por_coincidencia_de_tokens()

    corpus_tokenizado_seguro = [tokens or ["_"] for tokens in corpus_tokenizado]
    bm25 = BM25Okapi(corpus_tokenizado_seguro)
    scores = bm25.get_scores(query_tokens)

    orden = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    resultados = [
        ResultadoBusqueda(id_reporte=ids[i], texto=textos[i], score=float(scores[i]))
        for i in orden
        if scores[i] > 0
    ]
    if not resultados:
        # BM25 no encontró nada, pero puede ser el empate degenerado de IDF (término en
        # exactamente la mitad del corpus, o casos similares) y no una ausencia real del
        # término -- se intenta el fallback antes de aceptar "no hay resultados".
        return _por_coincidencia_de_tokens()
    return resultados[:top_n]
