# Archivo: app/api/routes/nlp.py
# Carpeta: microservicioMLL/app/api/routes/

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verificar_api_key
from NLP.clasificar_texto import ClasificadorTexto

# Endpoints de PLN que no encajan en history.py (que es sobre datos YA existentes de un lote:
# alertas, predicciones, recomendaciones, reportes) -- este archivo es para texto libre que
# alguien manda desde fuera del pipeline de sensores. Misma protección con X-Internal-Api-Key
# que el resto del microservicio.
router = APIRouter(tags=["nlp"], dependencies=[Depends(verificar_api_key)])

# Una sola instancia, cargada perezosamente en el primer clasificar() -- mismo criterio que
# `predictor = Predictor()` en app/api/routes/inference.py.
clasificador = ClasificadorTexto()


class ClasificarTextoRequest(BaseModel):
    id_usuario: int = Field(
        description="Usuario que envía el texto -- solo para bitácora/consistencia con el resto "
        "de la API; este endpoint no lee ni escribe datos de ningún lote, así que no hay nada "
        "que aislar por dueño aquí."
    )
    texto: str = Field(min_length=1, description="Texto libre a clasificar, ej. notas del productor")


@router.post("/nlp/clasificar-texto")
def clasificar_texto_endpoint(request: ClasificarTextoRequest):
    """Paso 4 de la opción B (clasificador de texto ligero, ver NLP/README.md): TF-IDF +
    Multinomial Naive Bayes (paso 2, `NLP/entrenar_clasificador_texto.py`) sugiere una severidad
    (`alta`/`critica`) para un texto libre en español.

    Pensado para texto que las reglas de dominio (`app/services/rules.py`) todavía no cubren --
    ej. una nota que el productor escriba con sus propias palabras (hoy no existe ese campo en
    el esquema, ver NLP/preparar_datos_clasificador.py). Los mensajes de alerta que el sistema
    ya genera NO necesitan pasar por aquí: su severidad ya se conoce exacta, por la regla que
    los generó.

    Si el clasificador todavía no está entrenado (pocas alertas reales acumuladas) o el texto
    viene vacío, responde `disponible: false` en vez de un error -- mismo criterio defensivo que
    el resto del proyecto usa para artefactos de ML que pueden no existir todavía."""
    resultado = clasificador.clasificar(request.texto)
    if resultado is None:
        return {
            "disponible": False,
            "mensaje": (
                "El clasificador todavía no está disponible (no hay suficientes alertas reales "
                "acumuladas para entrenarlo, o el texto enviado está vacío)."
            ),
        }
    return {"disponible": True, **resultado}
