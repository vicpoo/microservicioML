#NLP/clasificar_texto.py
"""
NLP/clasificar_texto.py

Paso 3 de la opción B: sirve en producción el clasificador entrenado en el paso 2
(`entrenar_clasificador_texto.py`). Mismo patrón que `app/services/predictor.py` en `ML/`: carga
perezosa del artefacto (`joblib.load` una sola vez), y si el artefacto no existe todavía --
porque no hay suficientes alertas reales acumuladas, ver
`NLP/preparar_datos_clasificador.py::MIN_EJEMPLOS_ENTRENAMIENTO` -- `clasificar()` devuelve
`None` en vez de tronar, para que quien lo llame (el endpoint del paso 4) pueda responder algo
razonable ("todavía no disponible") en vez de un error 500.

Qué recibe y qué regresa: texto libre en español (no necesariamente uno de los mensajes de
plantilla que ya conoce el sistema) -> severidad sugerida (`alta`/`critica`, las mismas dos
clases reales de `alertas.nivel_severidad` hoy) + qué tan seguro está el modelo de esa
predicción. Pensado para el día en que exista texto libre genuino que clasificar (ver el aviso
en `preparar_datos_clasificador.py`) -- con los mensajes de plantilla actuales, cualquier
severidad ya se sabe sin necesitar esto.
"""
import logging
import os
from typing import Dict, Optional

import joblib

logger = logging.getLogger(__name__)

RUTA_MODELO_DEFAULT = os.path.join(os.path.dirname(__file__), "artifacts", "clasificador_texto.joblib")


class ClasificadorTexto:
    """Carga perezosa: no toca disco en `__init__`, solo en el primer `clasificar()` -- así
    instanciarlo al importar el módulo (como hace `inference.py` con `Predictor()`) no falla ni
    tarda si el artefacto no existe todavía."""

    def __init__(self, ruta_modelo: str = RUTA_MODELO_DEFAULT):
        self.ruta_modelo = ruta_modelo
        self._intentado = False
        self._modelo = None

    def _cargar_si_hace_falta(self) -> None:
        if self._intentado:
            return
        self._intentado = True
        if not os.path.exists(self.ruta_modelo):
            return
        try:
            data = joblib.load(self.ruta_modelo)
            self._modelo = data["modelo"]
        except Exception as exc:
            logger.warning(f"[clasificar_texto] No se pudo cargar {self.ruta_modelo}: {exc}")
            self._modelo = None

    def disponible(self) -> bool:
        self._cargar_si_hace_falta()
        return self._modelo is not None

    def clasificar(self, texto: str) -> Optional[Dict]:
        """Devuelve `{"severidad_sugerida", "confianza", "probabilidades"}`, o `None` si el
        modelo no está disponible todavía o el texto viene vacío (nada que clasificar)."""
        self._cargar_si_hace_falta()
        if self._modelo is None:
            return None
        if not texto or not texto.strip():
            return None

        try:
            proba = self._modelo.predict_proba([texto])[0]
            clases = self._modelo.classes_
            idx = int(proba.argmax())
            return {
                "severidad_sugerida": str(clases[idx]),
                "confianza": round(float(proba[idx]) * 100, 1),
                "probabilidades": {str(c): round(float(p) * 100, 1) for c, p in zip(clases, proba)},
            }
        except Exception as exc:
            # Mismo criterio defensivo que Predictor.predecir(): un artefacto incompatible o un
            # error de predicción puntual no debe tumbar a quien llama.
            logger.warning(f"[clasificar_texto] Error al clasificar texto: {exc}")
            return None
