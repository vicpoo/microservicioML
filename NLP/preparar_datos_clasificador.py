#NLP/preparar_datos_clasificador.py
"""
NLP/preparar_datos_clasificador.py

Paso 1 de la opción B: recolección de datos de entrenamiento para el clasificador de texto
ligero. Mismo principio de responsabilidad única que ya sigue el resto del proyecto -- separar
"quién junta los datos" (este módulo) de "quién entrena" (paso 2, pendiente) y de "quién sirve
la predicción en producción" (paso 3, pendiente); el mismo patrón que
`scripts/recolectar_datos_reales.py` (recolecta) vs `scripts/train_models.py` (entrena) vs
`app/services/predictor.py` (sirve) en `ML/`.

Algoritmo elegido para esta opción (decisión de este paso): Multinomial Naive Bayes sobre
vectores TF-IDF. Se prefiere sobre KNN -- que es lo que ya usa el compañero para su parte de
PLN -- para no duplicar exactamente la misma técnica dentro del mismo proyecto: Naive Bayes es
el estándar clásico para clasificación de texto corto, no necesita normalizar distancias como
KNN, entrena instantáneo incluso con pocos ejemplos, y da una probabilidad por clase (no solo
"el vecino más parecido dice X") -- útil para decidir después, en el paso 3, si vale la pena
mostrar una predicción de baja confianza o no.

Aviso importante sobre los datos, léase antes de entrenar (paso 2): los mensajes de alerta que
existen hoy en `alertas.mensaje` salen de plantillas FIJAS en `app/services/rules.py` -- cada
tipo de alerta tiene EXACTAMENTE un texto, siempre igual (ver RECOMENDACIONES/MENSAJES_SEVERIDAD).
Un clasificador entrenado sobre estos mismos mensajes va a acertar prácticamente 100%, porque
memoriza strings exactos, no porque esté generalizando un patrón de lenguaje real -- ese
resultado no debe leerse como "el modelo es excelente". El valor real de este ejercicio es dejar
la piscina lista (vectorizador + modelo ya entrenado y servible) para el día en que exista texto
libre genuino que clasificar (ej. un campo de notas que el productor escriba con sus propias
palabras -- hoy no existe en el esquema, ver `app/models/lotes_cafe.py`); ahí sí habría señal
real que aprender, sin tener que escribir una regla nueva para cada frase posible que alguien
pudiera escribir.

También por eso `nivel_severidad` en la práctica solo toma dos valores reales hoy (`alta`,
`critica`): solo las alertas con severidad `riesgo`/`critico` del motor de reglas llegan a esta
tabla (`app/api/routes/inference.py::SEVERIDADES_QUE_ALERTAN`), y `notifier._SEVERIDAD_A_NIVEL`
las mapea a esas dos. No es un error de este módulo -- es el dato real tal como existe hoy.
"""
from typing import List, Tuple

from sqlalchemy.orm import Session

from app.models.alertas import Alerta

# Menos ejemplos que esto y ni vale la pena entrenar (el modelo terminaría memorizando 2-3
# frases sueltas, sin ninguna señal real que aprender) -- mismo criterio de umbral mínimo que
# usan ML/monitoreo.py y scripts/train_models.py para decidir si conviene (re)entrenar un
# artefacto en particular.
MIN_EJEMPLOS_ENTRENAMIENTO = 10


def recolectar_ejemplos(db: Session) -> Tuple[List[str], List[str]]:
    """Devuelve `(textos, etiquetas)` a partir de `alertas.mensaje` / `alertas.nivel_severidad`
    -- las mismas dos columnas que el resto del sistema ya escribe en producción, sin ninguna
    tabla ni dato nuevo. Descarta filas sin mensaje (no debería haberlas en la práctica, ya que
    `notifier.registrar_alerta` siempre pasa un mensaje no vacío, pero `Alerta.mensaje` es
    nullable en el esquema -- se filtra por si acaso, mismo criterio defensivo que el resto del
    proyecto en vez de asumir que el dato siempre viene completo)."""
    filas = (
        db.query(Alerta.mensaje, Alerta.nivel_severidad)
        .filter(Alerta.mensaje.isnot(None))
        .all()
    )
    textos = [mensaje for mensaje, _ in filas]
    etiquetas = [severidad for _, severidad in filas]
    return textos, etiquetas


def hay_suficientes_datos(db: Session) -> bool:
    """Punto de decisión que usará el paso 2 (entrenamiento): si no hay al menos
    `MIN_EJEMPLOS_ENTRENAMIENTO` mensajes reales, no se entrena -- mismo criterio defensivo que
    `scripts/train_models.py` ya usa para `rf_calidad`/`rf_tiempo_restante` (saltar el
    entrenamiento con un aviso claro, en vez de producir un modelo que memorizó 3 ejemplos y
    parece "funcionar" sin generalizar nada)."""
    textos, _ = recolectar_ejemplos(db)
    return len(textos) >= MIN_EJEMPLOS_ENTRENAMIENTO
