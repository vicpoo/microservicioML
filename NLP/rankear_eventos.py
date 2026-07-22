#NLP/rankear_eventos.py
"""
NLP/rankear_eventos.py

Nivel 2 del mini-pipeline de PLN: resumen extractivo con BM25 -- el modelo que faltaba más
allá de las plantillas del Nivel 1 (`generar_reporte.py`).

Cuándo se activa: cuando un lote acumuló muchas alertas (`UMBRAL_MUCHOS_EVENTOS` o más), el
reporte del Nivel 1 solo cuenta cuántas hubo de cada tipo -- no dice CUÁLES fueron las más
importantes. Este módulo puntúa cada mensaje de alerta con BM25 contra un "query" de términos
de urgencia y selecciona los más relevantes: un resumen EXTRACTIVO real (se seleccionan
oraciones que ya existen, escritas por `app/services/rules.py` vía `inference.py`, no se
inventa texto nuevo) en vez de simplemente listar las N más recientes.

Por qué BM25 y no BM42: BM42 (Qdrant, 2024) necesita una base de datos vectorial + un modelo de
embeddings adicional -- infraestructura que no tiene sentido para un servicio de este tamaño.
BM25 es puro Python (librería `rank_bm25`), sin GPU ni servicios externos, bien documentado
(es EL algoritmo clásico de recuperación de información -- lo usan buscadores desde los 90) y
fácil de auditar: mismo criterio que ya se usó para elegir un Algoritmo Genético interpretable
en vez de una red neuronal para la predicción de lluvia (ver ML/prediccion_lluvia_ga.py) -- se
prefiere un modelo simple y explicable sobre uno más "de moda" pero opaco o pesado de operar.

Cómo funciona BM25 aquí, en una frase: cada mensaje de alerta es un "documento", el query es un
puñado de palabras que señalan urgencia/severidad, y BM25 puntúa cada documento por cuánto se
parece a ese query (con la misma lógica que un buscador rankea páginas web contra lo que
escribiste) -- los mensajes con más términos de urgencia relevantes quedan primero.

El tokenizador y las stopwords viven en `NLP/texto_utils.py`, compartidos con
`NLP/buscar_reportes.py` (el buscador de historial completo, que usa el mismo BM25 pero sobre
otro corpus) -- una sola fuente de verdad para cómo se "leen" las palabras en todo el PLN de
este proyecto.
"""
from typing import List

from rank_bm25 import BM25Okapi

from NLP.texto_utils import tokenizar

# Debajo de este número de alertas, listar el conteo (Nivel 1) ya es suficientemente claro --
# BM25 solo aporta cuando hay demasiados eventos para mencionarlos todos.
UMBRAL_MUCHOS_EVENTOS = 5

# Cuántos mensajes destacados se devuelven cuando sí se activa el resumen extractivo.
TOP_N_DESTACADOS = 3

# Query de BM25: términos que señalan urgencia/severidad en el vocabulario que ya usan los
# mensajes de alerta reales (ver app/api/routes/inference.py::MENSAJES_SEVERIDAD y
# app/services/rules.py::RECOMENDACIONES) -- no es una lista arbitraria, son las palabras que
# de verdad aparecen en los mensajes más graves de este proyecto.
_QUERY_URGENCIA = (
    "critico urgente inmediata riesgo peligro atencion excesiva exceso lluvia detectada "
    "estancado brusco imposible"
)


def eventos_destacados(mensajes: List[str], top_n: int = TOP_N_DESTACADOS) -> List[str]:
    """Recibe los mensajes de alerta (ya sin filtrar por relevancia) y devuelve los `top_n`
    más relevantes según BM25, ordenados de más a menos relevante. Si hay `top_n` mensajes
    únicos o menos, los devuelve todos tal cual (rankear no aporta nada con tan pocos)."""
    unicos = list(dict.fromkeys(mensajes))  # dedup preservando el primer orden de aparición
    if len(unicos) <= top_n:
        return unicos

    corpus_tokenizado = [tokenizar(m) for m in unicos]
    # Documentos que quedaron sin ningún token útil (mensaje vacío o solo stopwords) no
    # deberían tumbar BM25Okapi -- se les da un token dummy para que sigan siendo parte del
    # corpus (con score bajo, no van a rankear alto de todas formas).
    corpus_tokenizado = [tokens or ["_"] for tokens in corpus_tokenizado]

    bm25 = BM25Okapi(corpus_tokenizado)
    scores = bm25.get_scores(tokenizar(_QUERY_URGENCIA))

    orden = sorted(range(len(unicos)), key=lambda i: scores[i], reverse=True)
    return [unicos[i] for i in orden[:top_n]]
