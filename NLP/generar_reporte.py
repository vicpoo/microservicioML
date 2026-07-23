#NLP/generar_reporte.py
"""
NLP/generar_reporte.py

Paso 2 del mini-pipeline de PLN: Generación de Lenguaje Natural (NLG) — convierte la
`DatosReporteLote` que arma el paso 1 (`recopilar_datos_reporte.py`) en un texto en español,
legible por una persona, sin ningún dato nuevo ni llamada a la BD (este módulo no conoce
SQLAlchemy ni sesiones de BD a propósito: recibe la estructura ya armada y solo redacta).

Nivel 1 de NLG (basado en reglas/plantillas): cada sección del reporte se arma con lógica
condicional sobre los datos reales -- no hay texto "inventado" ni generación probabilística,
así que el resultado es 100% predecible y auditable, igual que las recomendaciones de
`app/services/rules.py`. Es una extensión natural de esa misma idea: convertir datos
estructurados en texto, no una caja negra.

Estructura del reporte (siempre en este orden, cada sección se omite limpiamente si no hay
datos para ella en vez de inventar algo):
  1. Introducción: identifica el lote y cuánto lleva (o cuánto duró, si ya terminó) en secado.
  2. Alertas: cuántas hubo, de qué tipo (usando los mismos títulos cortos de
     `rules.TITULOS_CORTOS` que ya usa el push de FCM, para no tener una tercera redacción de
     los mismos tipos), y el estado de la más reciente.
  3. Riesgo de lluvia: lo que dice el Algoritmo Genético (`app/services/rain_predictor.py`),
     si hay una predicción reciente.
  4. Predicción de tiempo/calidad: si el modelo ya pudo calcularlas (hoy normalmente no, ver
     ML/README.md paso 9 -- se dice explícitamente que faltan datos, no se calla el hueco).
  5. Recomendaciones activas: la lista ya redactada por `app/services/recommender.py`.
  6. Cierre: una línea de resumen, más urgente si la severidad más alta vista fue crítica.
"""
from typing import Optional

from NLP.recopilar_datos_reporte import DatosReporteLote

# Mismo orden de severidad que app/services/notifier.py::SEVERITY_RANK, pero con las 4 llaves
# que de verdad aparecen en Alerta.nivel_severidad (baja/media/alta/critica) -- son escalas
# distintas (la de aquí es la del ENUM de Postgres, la de notifier es la de 4 niveles del
# motor de reglas) y no hay que confundirlas.
_RANK_SEVERIDAD_ALERTA = {"baja": 0, "media": 1, "alta": 2, "critica": 3}


def _formatear_horas(horas: float) -> str:
    horas = max(horas, 0.0)
    h = int(horas)
    m = round((horas - h) * 60)
    if m == 60:
        h, m = h + 1, 0
    if h == 0:
        return f"{m} minutos"
    if m == 0:
        return f"{h} hora{'s' if h != 1 else ''}"
    return f"{h} hora{'s' if h != 1 else ''} y {m} minutos"


def _seccion_introduccion(datos: DatosReporteLote) -> str:
    tipo = datos.tipo_proceso
    if datos.fecha_fin_secado is not None and datos.fecha_inicio_secado is not None:
        duracion = (datos.fecha_fin_secado - datos.fecha_inicio_secado).total_seconds() / 3600.0
        return (
            f"El lote {datos.nombre_lote} (proceso {tipo}) terminó su secado; "
            f"duró {_formatear_horas(duracion)} en total."
        )
    if datos.fecha_inicio_secado is None:
        return f"El lote {datos.nombre_lote} (proceso {tipo}) todavía no tiene fecha de inicio de secado registrada."
    return (
        f"El lote {datos.nombre_lote} (proceso {tipo}) lleva {_formatear_horas(datos.horas_transcurridas)} "
        "en secado."
    )


def _seccion_alertas(datos: DatosReporteLote) -> str:
    from app.services.rules import titulo_corto_para  # import diferido: evita que este módulo,
    # que es puro texto, dependa de app/ al importarse -- solo lo necesita para esta sección.
    from NLP.rankear_eventos import UMBRAL_MUCHOS_EVENTOS, eventos_destacados

    if datos.total_alertas == 0:
        return "No se han registrado alertas durante este proceso."

    partes_tipo = [
        f"{conteo} de {titulo_corto_para(tipo).lower()}"
        for tipo, conteo in sorted(datos.alertas_por_tipo.items(), key=lambda kv: -kv[1])
    ]
    resumen_tipos = ", ".join(partes_tipo)
    plural = "alertas" if datos.total_alertas != 1 else "alerta"
    texto = f"Se registraron {datos.total_alertas} {plural}: {resumen_tipos}."

    if datos.ultima_alerta is not None:
        estado = "ya fue atendida" if datos.ultima_alerta.atendida else "sigue sin atenderse"
        texto += (
            f" La más reciente fue de {titulo_corto_para(datos.ultima_alerta.tipo).lower()} "
            f"y {estado}."
        )

    # Nivel 2 (BM25, ver NLP/rankear_eventos.py): con pocas alertas, el conteo de arriba ya es
    # suficiente -- BM25 solo aporta cuando hay demasiados eventos para mencionarlos todos y
    # hace falta elegir cuáles destacar.
    if datos.total_alertas > UMBRAL_MUCHOS_EVENTOS and datos.mensajes_alertas:
        destacados = eventos_destacados(datos.mensajes_alertas)
        if destacados:
            lista = " ".join(f"({i}) {m}" for i, m in enumerate(destacados, start=1))
            texto += f" Eventos más relevantes según el modelo: {lista}"
    return texto


def _seccion_riesgo_lluvia(datos: DatosReporteLote) -> Optional[str]:
    pred = datos.ultima_prediccion
    if pred is None or pred.riesgo_lluvia_proxima is None:
        return None
    horas = pred.horas_anticipacion_lluvia or 3
    if pred.riesgo_lluvia_proxima:
        return f"El modelo estima riesgo de lluvia en las próximas {horas} horas: conviene cubrir el lote preventivamente."
    return f"El modelo no estima riesgo de lluvia en las próximas {horas} horas."


def _seccion_prediccion_calidad_tiempo(datos: DatosReporteLote) -> str:
    pred = datos.ultima_prediccion
    tiene_tiempo = pred is not None and pred.tiempo_estimado_horas is not None
    tiene_calidad = pred is not None and pred.calidad_estimada is not None

    if not tiene_tiempo and not tiene_calidad:
        return "No hay suficientes datos históricos todavía para estimar con precisión el tiempo restante ni la calidad final."

    partes = []
    if tiene_tiempo:
        partes.append(f"un tiempo restante estimado de {_formatear_horas(pred.tiempo_estimado_horas)}")
    if tiene_calidad:
        confianza_txt = f" (confianza {pred.confianza:.0f}%)" if pred.confianza is not None else ""
        # calidad_estimada es un puntaje escala SCA 0-100 (ver migration.sql paso 10), no una
        # categoría -- es una aproximación basada en condiciones de secado, no una catación real.
        partes.append(f"una calidad final estimada de {pred.calidad_estimada:.0f}/100 en escala SCA{confianza_txt}")
    return "El modelo predice " + " y ".join(partes) + "."


def _seccion_recomendaciones(datos: DatosReporteLote) -> Optional[str]:
    if not datos.recomendaciones_activas:
        return None
    if len(datos.recomendaciones_activas) == 1:
        return f"Recomendación: {datos.recomendaciones_activas[0]}"
    lista = " ".join(f"({i}) {texto}" for i, texto in enumerate(datos.recomendaciones_activas, start=1))
    return f"Recomendaciones: {lista}"


def _seccion_cierre(datos: DatosReporteLote) -> str:
    if not datos.alertas_por_severidad:
        return "El lote se encuentra dentro de los parámetros esperados."
    peor = max(datos.alertas_por_severidad, key=lambda sev: _RANK_SEVERIDAD_ALERTA.get(sev, 0))
    if peor == "critica":
        return "Se recomienda revisar el lote cuanto antes por la severidad de las alertas registradas."
    if peor == "alta":
        return "Conviene revisar el lote pronto."
    return "El seguimiento normal es suficiente por ahora."


def generar_reporte_lote(datos: DatosReporteLote) -> str:
    """Punto de entrada del paso 2. Recibe la estructura del paso 1 y devuelve el reporte
    completo como un solo string en español, listo para mostrarse tal cual (ej. en la app
    móvil o en GET /anomalies/{id_lote}/reporte, paso 3, pendiente)."""
    secciones = [
        _seccion_introduccion(datos),
        _seccion_alertas(datos),
        _seccion_riesgo_lluvia(datos),
        _seccion_prediccion_calidad_tiempo(datos),
        _seccion_recomendaciones(datos),
        _seccion_cierre(datos),
    ]
    return " ".join(s for s in secciones if s)
