#NLP/entrenar_clasificador_texto.py
"""
NLP/entrenar_clasificador_texto.py

Paso 2 de la opción B: entrena el clasificador de texto ligero (TF-IDF + Multinomial Naive
Bayes, decisión tomada en el paso 1 -- ver `NLP/preparar_datos_clasificador.py`) sobre los datos
reales que ya junta ese módulo, y guarda el artefacto entrenado en `NLP/artifacts/`. Mismo
patrón que `scripts/train_models.py` en `ML/`: recolectar (paso 1) -> entrenar (este módulo) ->
servir en producción (paso 3, `clasificar_texto.py`, pendiente), cada responsabilidad en su
propio archivo.

El vectorizador (`TfidfVectorizer`) usa el MISMO tokenizador que ya comparten
`rankear_eventos.py` y `buscar_reportes.py` (`NLP/texto_utils.py::tokenizar`) -- una sola forma
de leer palabras en todo el PLN del proyecto, ahora también para el paso de vectorización.

Recordatorio del aviso de `preparar_datos_clasificador.py`: con los mensajes de alerta actuales
(texto de plantilla fijo por tipo), la exactitud sobre el propio set de entrenamiento va a salir
muy alta -- es memorización de strings, no evidencia de que el modelo "aprendió español". Se
reporta igual en las métricas (transparencia), pero no debe leerse como el modelo siendo
excepcionalmente bueno.
"""
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.models.database import SessionLocal, init_db
from NLP.preparar_datos_clasificador import hay_suficientes_datos, recolectar_ejemplos
from NLP.texto_utils import tokenizar

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
RUTA_MODELO = os.path.join(ARTIFACTS_DIR, "clasificador_texto.joblib")
RUTA_METRICAS = os.path.join(ARTIFACTS_DIR, "metricas_clasificador_texto.json")


def _construir_pipeline() -> Pipeline:
    return Pipeline([
        # tokenizer propio (compartido con el resto del PLN) -> lowercase=False y
        # token_pattern=None para que TfidfVectorizer no aplique SU preprocesamiento por
        # default encima del nuestro (evitaría, por ejemplo, aplicar min_df/stopwords propias
        # que no coinciden con NLP/texto_utils.py).
        ("tfidf", TfidfVectorizer(tokenizer=tokenizar, lowercase=False, token_pattern=None)),
        ("clf", MultinomialNB()),
    ])


def entrenar(
    db: Session, ruta_modelo: str = RUTA_MODELO, ruta_metricas: str = RUTA_METRICAS
) -> Optional[dict]:
    """Entrena y guarda el artefacto si hay suficientes datos reales (paso 1). Si no los hay,
    NO entrena -- mismo criterio defensivo que `scripts/train_models.py` usa para
    `rf_calidad`/`rf_tiempo_restante`: es mejor no tener el artefacto todavía que tener uno
    entrenado con 3 ejemplos que parece funcionar pero no generaliza nada. Devuelve las métricas
    del entrenamiento, o `None` si se saltó por falta de datos.

    `ruta_modelo`/`ruta_metricas` son parametrizables (con los mismos valores de producción por
    default) para que las pruebas (paso 5) puedan entrenar hacia un archivo temporal sin tocar
    el artefacto real del proyecto -- mismo motivo por el que las pruebas usan una BD sqlite
    aparte en vez de la real."""
    if not hay_suficientes_datos(db):
        return None

    textos, etiquetas = recolectar_ejemplos(db)
    pipeline = _construir_pipeline()
    pipeline.fit(textos, etiquetas)

    # Exactitud sobre el propio set de entrenamiento (no hay suficientes datos reales todavía
    # para separar un test aparte con sentido, ver aviso del módulo) -- se reporta de todas
    # formas por transparencia, aclarando en las métricas que es "train", no "test".
    predicciones = pipeline.predict(textos)
    exactitud_train = float(np.mean(np.array(predicciones) == np.array(etiquetas)))

    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
    joblib.dump({"modelo": pipeline, "clases": sorted(pipeline.classes_.tolist())}, ruta_modelo)

    metricas = {
        "n_ejemplos": len(textos),
        "clases": sorted(set(etiquetas)),
        "exactitud_train": round(exactitud_train, 4),
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "aviso": (
            "Exactitud calculada sobre el mismo set de entrenamiento, con mensajes de plantilla "
            "fija -- una exactitud alta aquí es esperada (memorización), no evidencia de "
            "generalización real. Ver NLP/preparar_datos_clasificador.py."
        ),
    }
    os.makedirs(os.path.dirname(ruta_metricas), exist_ok=True)
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    return metricas


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        resultado = entrenar(db)
        if resultado is None:
            textos, _ = recolectar_ejemplos(db)
            print(
                f"[entrenar_clasificador_texto] Solo hay {len(textos)} mensajes de alerta reales "
                "todavía -- no se entrenó (ver MIN_EJEMPLOS_ENTRENAMIENTO en "
                "NLP/preparar_datos_clasificador.py). Reintenta cuando se acumulen más alertas."
            )
            return 1
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
