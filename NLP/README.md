# NLP — kajve

Carpeta separada de `ML/` a propósito: `ML/` es el pipeline de Machine Learning "clásico"
(detección de anomalías, predicción de tiempo/calidad/lluvia — ver `ML/README.md`, pasos 1-12).
`NLP/` es la capa de Procesamiento de Lenguaje Natural: toma la salida de ese pipeline (alertas,
predicciones, recomendaciones) y la convierte en texto en lenguaje natural — un reporte legible
por lote, no un chatbot ni nada conversacional.

| Paso | Archivo | Entrada | Salida |
|---|---|---|---|
| 1. Recopilación de datos del reporte | `recopilar_datos_reporte.py` | tablas `lotes_cafe`, `alertas`, `predicciones`, `recomendaciones` (en vivo) | `DatosReporteLote` (dataclass estructurada, sin texto en lenguaje natural todavía) |
| 2. Generación de texto (NLG) | `generar_reporte.py` | `DatosReporteLote` (paso 1) | Texto en español, listo para mostrarse |
| 3. Endpoint | `app/api/routes/history.py::obtener_reporte` | pasos 1+2 | `GET /anomalies/{id_lote}/reporte` |
| 4. Historial | `registrar_reporte.py` + tabla `reportes_lote` | texto del paso 2 | `GET /anomalies/{id_lote}/reportes` (historial guardado) |
| A.1-2. Buscador de historial (BM25) | `texto_utils.py` + `buscar_reportes.py` | historial completo de reportes de un usuario | ranking de reportes relevantes a un query libre |
| A.3. Endpoint del buscador | `app/api/routes/history.py::buscar_reportes_endpoint` | paso A.1-2 | `GET /anomalies/reportes/buscar?id_usuario=&query=` |
| B.1. Datos del clasificador | `preparar_datos_clasificador.py` | tabla `alertas` (global, sin filtrar por usuario) | textos + etiquetas de severidad |
| B.2. Entrenamiento | `entrenar_clasificador_texto.py` | paso B.1 | `NLP/artifacts/clasificador_texto.joblib` (TF-IDF + Naive Bayes) |
| B.3. Servir en producción | `clasificar_texto.py` | paso B.2 | severidad sugerida + confianza para un texto libre |
| B.4. Endpoint del clasificador | `app/api/routes/nlp.py::clasificar_texto_endpoint` | paso B.3 | `POST /nlp/clasificar-texto` |

## Por qué una carpeta separada de `ML/`

Son dos técnicas de IA distintas resolviendo problemas distintos: `ML/` aprende patrones de
datos numéricos/categóricos de sensores (IsolationForest, RandomForest, Algoritmo Genético);
`NLP/` transforma esos resultados estructurados en lenguaje natural (Generación de Lenguaje
Natural / NLG). Mantenerlas en carpetas separadas dentro del mismo proyecto deja claro, para
quien revise el código, qué parte corresponde a cada técnica.

## Paso 1 — Recopilación de datos (`recopilar_datos_reporte.py`)

Responsabilidad única: reunir y estructurar los datos crudos de un lote — SIN redactar ni una
palabra de texto todavía. Esa parte (plantillas de redacción) es el paso 2, pendiente. Se
separan a propósito (mismo principio que `ML/entrenamiento.py` vs `ML/evaluacion.py`): cada
módulo hace una sola cosa, se puede probar aparte, y el paso 2 no necesita saber nada de SQL ni
de los modelos de SQLAlchemy.

`recopilar_datos_lote(db, id_lote) -> Optional[DatosReporteLote]`:
- Devuelve `None` si el lote no existe (el endpoint del paso 3 decide qué status HTTP usar).
- `DatosReporteLote` trae: datos básicos del lote (nombre, tipo de proceso, horas
  transcurridas — reutiliza `app/services/lectura_utils.py::calcular_horas_transcurridas`, no
  recalcula la fórmula por su cuenta), conteo de alertas por tipo y por severidad, la alerta
  más reciente, la predicción más reciente (tiempo estimado, calidad estimada, riesgo de
  lluvia), y hasta 5 recomendaciones activas más recientes sin texto duplicado.
- Probado con tres casos: lote inexistente (`None`), lote real sin ningún historial todavía
  (listas/diccionarios vacíos, nada revienta), y lote con alertas/predicción/recomendaciones
  reales (agregación y deduplicación correctas).

## Paso 2 — Generación de texto / NLG (`generar_reporte.py`)

`generar_reporte_lote(datos: DatosReporteLote) -> str`: recibe la estructura del paso 1 (nunca
toca la BD directamente) y arma el reporte en español, sección por sección, cada una con su
propia función y su propia lógica condicional:

1. **Introducción** — identifica el lote; distingue "lleva N horas en secado" (en proceso) de
   "terminó su secado; duró N horas" (ya finalizado, calculado con
   `fecha_fin_secado - fecha_inicio_secado`, no con el reloj actual) de "todavía no tiene fecha
   de inicio registrada" (lote recién creado).
2. **Alertas** — cuenta total + desglose por tipo, reutilizando `rules.titulo_corto_para()`
   (los mismos títulos de una línea que ya usa el push de FCM, para no mantener una tercera
   redacción de los mismos tipos) y si la más reciente ya fue atendida o no.
3. **Riesgo de lluvia** — lo que dijo el Algoritmo Genético en la predicción más reciente; se
   omite limpiamente si nunca se calculó.
4. **Predicción de tiempo/calidad** — si el modelo ya pudo calcularlas, o dice explícitamente
   que todavía no hay datos suficientes (no se calla el hueco, igual que el resto del proyecto).
5. **Recomendaciones activas** — reusa el texto que ya redactó `app/services/recommender.py`,
   no inventa uno nuevo.
6. **Cierre** — una línea de urgencia según la peor severidad de alerta vista.

Cada sección se omite (no se rellena con texto genérico) si no hay datos para ella. Probado con
5 casos: lote sin fecha de inicio ni historial, lote en proceso con alertas y predicciones
parciales, lote con predicción completa de tiempo/calidad y riesgo de lluvia en `True`, lote ya
finalizado, y lote inexistente (`None` desde el paso 1, este módulo ni se llama).

Es Nivel 1 de NLG (basado en reglas/plantillas): 100% determinístico y auditable, mismo
espíritu que las recomendaciones de `rules.py`. Sobre este módulo se monta el Nivel 2 (resumen
extractivo con BM25, ver sección propia más abajo) sin haber tenido que tocar el paso 1 más que
para agregar un campo de datos nuevo.

## Paso 3 — Endpoint (`GET /anomalies/{id_lote}/reporte`)

Vive en `app/api/routes/history.py`, junto a los otros GETs de historial, con el mismo
`_verificar_dueno()` -- 403 si el lote no es del `id_usuario` que llama, 404 si no existe.
Combina paso 1 + paso 2 al vuelo (nada se persiste todavía):

```
GET /api/v1/anomalies/{id_lote}/reporte?id_usuario=1

200 OK
{
  "id_lote": 1,
  "reporte_texto": "El lote ... lleva 10 horas en secado. Se registraron 1 alerta: ...",
  "fecha_generado": "2026-07-22T07:02:23.581327"
}
```

Probado con los 3 casos que importan para un endpoint con dueño: dueño real (200 + reporte),
otro usuario (403), lote inexistente (404).

## Paso 4 — Historial (`registrar_reporte.py` + tabla `reportes_lote`)

Cada llamada a `GET .../reporte` (singular) genera el texto al momento Y lo guarda en
`reportes_lote` (tabla nueva, sección 8 de `migration.sql`) -- mismo criterio que
`predicciones`/`alertas`/`recomendaciones`: se acumula historial, no se sobrescribe. Nuevo
endpoint `GET /anomalies/{id_lote}/reportes` (plural) para verlo, con el mismo
`_verificar_dueno()` de siempre.

```
GET /api/v1/anomalies/1/reporte?id_usuario=1     -> genera Y guarda, devuelve el texto de ahora
GET /api/v1/anomalies/1/reportes?id_usuario=1    -> lista los ya guardados, más reciente primero
```

Separación de responsabilidades explícita: `generar_reporte.py` (paso 2) es puro texto, no
toca la BD; `registrar_reporte.py` (paso 4) es puro I/O de BD, no redacta nada. Probado
generando 2 reportes seguidos de un mismo lote (con una alerta nueva entre medio, para
confirmar que el segundo reporte sí refleja el cambio) y verificando que el historial los lista
a ambos, más reciente primero, y que sigue dando 403 para otro usuario.

## Paso 5 — Pruebas formales (`tests/test_nlp.py`)

Antes cada paso se probó con scripts de humo sueltos (útiles mientras se construía, pero no
quedaban en el repo). `tests/test_nlp.py` los formaliza para que corran con
`pytest tests/ -q` junto al resto de la suite, en dos niveles:

- **Unitarias sobre `generar_reporte.py`**, con `DatosReporteLote` sintéticos (sin BD, rápidas):
  sin alertas ni predicción, alertas variadas (cuenta y nombres correctos, cierre con
  urgencia), lote finalizado (duración real, no el reloj actual), riesgo de lluvia en ambos
  sentidos, predicción completa de tiempo/calidad.
- **De integración sobre el endpoint**, con BD sqlite de prueba (mismo patrón que
  `tests/test_api.py`, en un rango de IDs separado para no pisarse): aislamiento por dueño
  (200/403/404) y que el historial se acumule reflejando cambios reales entre un reporte y el
  siguiente.

**Bug real que encontraron estas pruebas:** `historial_reportes()` ordenaba solo por
`fecha_generado DESC`; como esa columna tiene resolución de 1 segundo, dos reportes generados
casi seguido (exactamente lo que hace la prueba de integración, y lo que podría pasar en
producción con un cliente rápido) podían empatar y quedar en orden indefinido. Se corrigió
agregando `id_reporte DESC` como desempate.

## Nivel 2 — Resumen extractivo con BM25 (`rankear_eventos.py`)

Extensión opcional del paso 2, ya implementada: cuando un lote acumula muchas alertas
(`UMBRAL_MUCHOS_EVENTOS = 5` o más), el reporte del Nivel 1 solo puede decir "cuántas hubo de
cada tipo" — no cuáles importan más. `rankear_eventos.py` agrega un verdadero modelo de PLN
encima de las plantillas: **BM25** (librería `rank_bm25`), el algoritmo clásico de recuperación
de información (el mismo principio detrás de buscadores desde los años 90).

**Cómo funciona:** cada mensaje de alerta ya redactado (por `app/services/rules.py`, vía
`inference.py`) es un "documento". Se define un `query` fijo de términos de urgencia/severidad
tomados del vocabulario real del proyecto (`critico`, `riesgo`, `lluvia`, `estancado`,
`imposible`, etc.). BM25 puntúa cada mensaje por cuánto se parece a ese query y devuelve los
`TOP_N_DESTACADOS = 3` con mayor puntaje — un resumen **extractivo** real: se seleccionan
oraciones que ya existen, no se genera texto nuevo. Con `top_n` mensajes únicos o menos, no
tiene sentido rankear y se devuelven todos tal cual.

**Por qué BM25 y no BM42 (u otra opción "de moda"):**

- **BM42** (Qdrant, 2024) necesita una base de datos vectorial más un modelo de embeddings —
  infraestructura adicional (servicio externo, GPU/latencia, otro punto de falla) que no se
  justifica para reordenar unos cuantos mensajes de texto corto en un microservicio de este
  tamaño.
- **BM25** es puro Python (`rank_bm25`), corre en el mismo proceso sin dependencias externas ni
  descargas en tiempo de ejecución (a propósito no se usa el corpus de stopwords de NLTK, para
  no depender de una descarga al desplegar — mismo criterio que `app/services/fcm.py` de no
  depender de red externa para arrancar), y es trivial de auditar: se puede leer el `query` y
  entender exactamente por qué un mensaje quedó primero.
- **Mismo criterio que ya se usó en el proyecto**: para predecir riesgo de lluvia se eligió un
  Algoritmo Genético interpretable (`ML/prediccion_lluvia_ga.py`) en vez de una red neuronal de
  caja negra. Aquí se repite la misma decisión de diseño — preferir un modelo simple, explicable
  y barato de operar sobre uno más sofisticado pero opaco o pesado — porque el problema (ordenar
  ~50 mensajes de texto corto) no necesita más que eso.

**Integración:** `generar_reporte.py::_seccion_alertas()` llama a `eventos_destacados()` solo
cuando `total_alertas > UMBRAL_MUCHOS_EVENTOS` y hay `mensajes_alertas` disponibles (recolectados
en el paso 1 por `_mensajes_alertas()`, hasta `MAX_MENSAJES_ALERTA = 50` más recientes). Agrega
una frase al final de la sección de alertas: *"Eventos más relevantes según el modelo: (1) ...
(2) ... (3) ..."*. Con 5 alertas o menos, la sección no cambia — el conteo del Nivel 1 ya es
suficiente.

Probado en dos niveles: `eventos_destacados()` de forma aislada con 8 mensajes reales de
severidad mixta (confirmó que los 2 mensajes "Riesgo crítico" quedan primero, por encima de
mensajes de severidad leve), y `generar_reporte_lote()` de punta a punta con un `DatosReporteLote`
sintético de 6 alertas (activa la sección BM25 correctamente) y otro de 2 alertas (no la activa,
sin regresión al Nivel 1). Suite completa (`pytest tests/ -q`, 13 pruebas) sigue pasando después
del cambio en `DatosReporteLote` (campo nuevo `mensajes_alertas`).

## Opción A — Buscador de reportes históricos con BM25

Segunda pieza de PLN, separada del reporte de un lote individual: un buscador de texto libre
sobre TODO el historial de reportes ya generados (`reportes_lote`) de un usuario, a través de
todos sus lotes. Mismo algoritmo que el Nivel 2 (BM25), pero usado como lo que realmente es —
un motor de búsqueda — en vez de un resumen extractivo dentro de un solo reporte.

**Paso 1 — Tokenizador compartido (`NLP/texto_utils.py`).** El tokenizador y las stopwords que
antes vivían solo dentro de `rankear_eventos.py` se sacaron a un módulo aparte, para que el
resumen extractivo (Nivel 2) y el buscador usen exactamente la misma forma de leer palabras —
una sola fuente de verdad, mismo criterio que `app/services/rules.py` con los umbrales de
dominio.

**Paso 2 — `NLP/buscar_reportes.py`.** `buscar_reportes(corpus, query, top_n=5)` recibe el
historial ya armado (pares `id_reporte`/texto, sin tocar la BD -- eso lo hace el endpoint) y
regresa los reportes más relevantes al query. A diferencia del Nivel 2 (que rankea aunque los
documentos sean casi idénticos, y con pocos ni siquiera rankea), aquí sí importa el orden real
incluso con pocos documentos, y no se deduplica texto (dos reportes de lotes distintos con texto
idéntico son resultados válidos y distintos).

Al probarlo con datos reales aparecieron dos problemas genuinos de BM25 con corpus chicos —
justo el caso más común (un usuario que apenas empieza, con 1-2 reportes todavía):

- Con **1 solo documento**, el IDF de BM25 es matemáticamente degenerado (toda palabra "aparece
  en el 100% del corpus" por definición) y el algoritmo da scores negativos sin importar el
  query, aunque el texto sí contenga lo buscado.
- Con un término que aparece en **exactamente la mitad** de los reportes de un usuario (ej.
  "lluvia detectada" en 2 de 4), su IDF da matemáticamente 0 y el score queda en 0 en los cuatro
  documentos -- sin ser un bug de BM25 en sí (un término que no discrimina entre la mitad de tus
  documentos legítimamente no debería rankear alto), el resultado práctico es "no encontré nada"
  cuando sí había una coincidencia real.

Se resolvió con una red de seguridad de dos capas: con menos de `UMBRAL_CORPUS_CHICO = 3`
documentos, se usa directamente un conteo de tokens del query en común (sin BM25); y si BM25 sí
corrió pero no encontró ningún score positivo, se reintenta con ese mismo conteo antes de
aceptar "no hay resultados". Ambos casos quedan documentados en el docstring del módulo y
cubiertos por pruebas formales (ver paso 4).

**Paso 3 — Endpoint (`GET /anomalies/reportes/buscar`).** Vive en `history.py`, con una ruta
estática (no `/anomalies/{id_lote}/...`) porque busca sobre TODO el historial del usuario, no
sobre un lote puntual. El aislamiento aquí es más simple que `_verificar_dueno`: el corpus se
arma con un JOIN `reportes_lote` -> `lotes_cafe.id_usuario = id_usuario`, así que es
estructuralmente imposible que aparezca un reporte ajeno en los resultados.

**Paso 4 — Pruebas (`tests/test_nlp.py`).** Unitarias sobre `buscar_reportes()` (sin BD): corpus
o query vacío, el caso de 1 documento, el empate exacto de 50% (el bug real encontrado arriba),
BM25 normal con 6 documentos, sin relación real, y el límite de `top_n`. De integración sobre el
endpoint (con BD): dos usuarios con reportes propios, confirma que cada uno encuentra sus
reportes relevantes y nunca los del otro, y que un usuario sin historial o un query sin
coincidencia regresan una lista vacía (200, no error).

## Opción B — Clasificador de texto ligero (TF-IDF + Naive Bayes)

Tercera pieza de PLN: un clasificador supervisado que sugiere severidad (`alta`/`critica`) para
texto libre en español, a diferencia de BM25 (Nivel 2 y opción A) que es no supervisado
(rankea/busca, no clasifica). Con esto el proyecto tiene dos técnicas de PLN genuinamente
distintas, no solo dos usos del mismo algoritmo.

**Paso 1 — Datos y elección del modelo (`preparar_datos_clasificador.py`).** `recolectar_ejemplos(db)`
junta `alertas.mensaje`/`alertas.nivel_severidad` -- a propósito SIN filtrar por lote ni usuario
(el clasificador es global: aprende del lenguaje de las alertas en general, no de un usuario en
particular). Algoritmo elegido: **Multinomial Naive Bayes sobre TF-IDF**, no KNN (que es lo que
ya usa el compañero de equipo para su parte de PLN, así no se duplica la misma técnica dentro
del mismo proyecto) -- Naive Bayes es el estándar clásico para texto corto, entrena al instante
incluso con pocos ejemplos, y da una probabilidad por clase en vez de solo "el vecino más
parecido dice X".

**Aviso importante, documentado desde este paso:** los mensajes de `alertas.mensaje` salen de
plantillas FIJAS (`app/services/rules.py`) -- cada tipo de alerta tiene exactamente un texto,
siempre igual. Un clasificador entrenado sobre esos mismos mensajes memoriza strings exactos, no
generaliza un patrón real de lenguaje. El valor real de esta pieza es quedar lista para el día
en que exista texto libre genuino que clasificar (ej. notas del productor escritas con sus
propias palabras -- ese campo no existe todavía en el esquema).

**Paso 2 — Entrenamiento (`entrenar_clasificador_texto.py`).** `entrenar(db)` vectoriza con
`TfidfVectorizer` (mismo tokenizador compartido de `NLP/texto_utils.py` que ya usan BM25 y el
buscador -- una sola forma de leer palabras en todo el PLN) y entrena `MultinomialNB`. Si no hay
al menos `MIN_EJEMPLOS_ENTRENAMIENTO = 10` mensajes reales, NO entrena -- mismo criterio
defensivo que `scripts/train_models.py` usa para `rf_calidad`/`rf_tiempo_restante`. Corre como
script (`python3 -m NLP.entrenar_clasificador_texto`), guarda el artefacto en
`NLP/artifacts/clasificador_texto.joblib` + métricas en JSON.

Confirmado con datos sintéticos: con 0 alertas no entrena; con 15 (mensajes de plantilla) sí
entrena, con **exactitud sobre el propio set de entrenamiento de 1.0 exacta** -- la memorización
esperada, reportada tal cual en las métricas para no leerse como "modelo excelente". Con texto
libre real que el modelo nunca vio ("hay agua encima del café y está lloviendo mucho", "todo se
ve normal, sin problemas"), las probabilidades salieron cercanas a 50/50 (y hasta clasificó mal
el caso normal) -- confirma en la práctica la limitación documentada arriba: sin texto libre real
para entrenar, no hay generalización real todavía.

**Paso 3 — Servir en producción (`clasificar_texto.py`).** Clase `ClasificadorTexto` con carga
perezosa del artefacto (no toca disco al instanciarse, mismo patrón que `Predictor` en `ML/`).
`clasificar(texto) -> {"severidad_sugerida", "confianza", "probabilidades"}`, o `None` si el
artefacto no existe todavía o el texto viene vacío -- nunca truena.

**Paso 4 — Endpoint (`POST /nlp/clasificar-texto`).** Vive en un archivo nuevo,
`app/api/routes/nlp.py` (no en `history.py`, porque este endpoint no lee/escribe datos de
ningún lote -- no hay nada que aislar por dueño aquí, `id_usuario` es solo bitácora). Si el
clasificador no está disponible, responde `{"disponible": false, "mensaje": "..."}` en vez de un
error.

**Paso 5 — Pruebas (`tests/test_nlp.py`).** Estas pruebas usan una BD sqlite propia y AISLADA
(no la compartida `SessionLocal` del resto de la suite), porque `recolectar_ejemplos()` lee TODA
la tabla `alertas` sin filtrar -- usar la BD compartida mezclaría alertas sembradas por
`test_api.py` y por el resto de `test_nlp.py` con las de estas pruebas, volviendo no
deterministas las aserciones de conteo/clases. Cubren: `recolectar_ejemplos` ignora mensajes
nulos y respeta el umbral mínimo; `entrenar()` no crea artefacto sin datos suficientes y sí lo
crea (con métricas correctas) cuando hay suficientes; `ClasificadorTexto` sin artefacto regresa
`None` sin tronar y con artefacto clasifica texto libre con probabilidades válidas; y el endpoint
completo, tanto sin modelo disponible como con uno entrenado (usando `monkeypatch` para apuntar
el clasificador del endpoint a un artefacto de prueba, sin tocar nunca el artefacto real del
proyecto).

## Pendiente

- Nada bloqueante. Posible siguiente paso, no pedido todavía: exponer en el reporte cuál fue el
  "score" de BM25 de cada evento destacado, si en algún momento se quiere mostrar esa
  transparencia al usuario final.
