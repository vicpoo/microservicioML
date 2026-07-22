#NLP/texto_utils.py
"""
NLP/texto_utils.py

Utilidades de texto compartidas entre los distintos módulos de PLN que usan BM25
(`rankear_eventos.py`, resumen extractivo dentro de un reporte; `buscar_reportes.py`,
buscador sobre el historial completo de reportes de un usuario). Se separan aquí para que
ambos usen EXACTAMENTE el mismo tokenizador y la misma lista de stopwords -- mismo criterio
de "una sola fuente de verdad" que ya sigue `app/services/rules.py` con los umbrales de
dominio: si el tokenizador cambiara en un solo lugar y no en el otro, los dos "buscadores"
del proyecto empezarían a comportarse distinto sin ninguna razón real.
"""
import re
from typing import List

# Lista corta de stopwords en español -- a propósito NO se usa el corpus de NLTK aquí (evita
# depender de una descarga de datos en tiempo de ejecución/despliegue; ver
# app/services/fcm.py::_obtener_app para el mismo criterio de "que no dependa de red externa
# para arrancar"). Cubre los conectores más comunes que aparecen en los mensajes reales del
# proyecto (alertas, recomendaciones, reportes NLG).
STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "en",
    "y", "o", "que", "se", "su", "sus", "es", "son", "por", "para", "con", "sin", "no",
    "lo", "le", "les", "este", "esta", "estos", "estas", "ya", "sobre", "muy",
}

_TOKEN_RE = re.compile(r"[a-záéíóúñü]+")


def tokenizar(texto: str) -> List[str]:
    """Minúsculas, separa en palabras (solo letras, con acentos/ñ), descarta stopwords y
    palabras de 2 letras o menos (artículos sueltos, conectores cortos que ya no cubre la
    lista de arriba). Usada tanto para indexar documentos como para el query de BM25 -- debe
    ser la MISMA función en ambos lados, o el índice y el query terminan hablando "idiomas"
    de tokens distintos."""
    texto = texto.lower()
    tokens = _TOKEN_RE.findall(texto)
    return [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]
