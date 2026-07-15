# Análisis: qué le sirve a microservicioMLL del ML de riesgo prenatal

Comparación entre el `machine_learning_service` del otro equipo (estratificación de riesgo prenatal: K-Means + KNN + NLP clínico) y nuestro `microservicioMLL` (calidad del café + tiempo de secado + recomendaciones/alertas + detección de anomalías), con recomendaciones concretas de qué adoptar, qué no, y por qué.

## 1. Resumen ejecutivo — los dos proyectos lado a lado

| | Riesgo prenatal (otro equipo) | microservicioMLL (nuestro) |
|---|---|---|
| **Objetivo** | Asignar a una paciente a 1 de 4 perfiles de riesgo prenatal | Predecir calidad final, tiempo restante, recomendar acciones y detectar anomalías en el secado |
| **Naturaleza del problema** | Descubrir taxonomía clínica que no existía de antemano | Las categorías ya existen (calidad: Cuadro 8/11; severidad: Cuadro 9); no hay que "descubrirlas" |
| **No supervisado** | K-Means (descubre 4 perfiles), validado con Elbow/Silhouette/DBSCAN/Jerárquico | IsolationForest (detecta outliers, no agrupa) |
| **Supervisado** | KNN (destila las etiquetas de K-Means para producción) | RandomForest ×3 (tipo de anomalía, tiempo restante, calidad final) |
| **Motor de reglas** | No tiene; todo pasa por el modelo | Sí — reglas de dominio (`rules.py`) son la fuente autoritativa, el ML complementa |
| **Explicabilidad (XAI)** | Sí — módulo dedicado (`explainability.py`): afinidad, factores determinantes, casos límítrofes, pacientes similares, narrativa | No tiene; solo devuelve `variables_contribuyentes` (lista plana, sin ranking ni z-scores) |
| **Detección de atípicos vs. población** | Sí — compara paciente contra centroide de su clúster (`model_metadata.json`) | No tiene; solo detecta atípicos vía IsolationForest (caja negra) o reglas de umbral fijo |
| **Registro/versionado de modelo** | Tabla `ml_models` con `is_active`, `metadata_config` JSONB, `seed_model.py` | Tabla `modelos_ml` con lógica similar (`get_or_create_modelo`), pero sin metadata rica (hiperparámetros/centroides) |
| **Validación del entrenamiento** | Elbow, Silhouette, ARI contra ground truth sintético | `GroupShuffleSplit` por lote (evita fuga de datos) + accuracy/f1/rmse/mae |
| **NLP** | Sí — extrae síntomas y zonas corporales de texto libre | No aplica (no hay texto libre en el dominio actual) |
| **Testing** | pytest con fixtures que aíslan la BD por transacción/savepoint | pytest con `TestClient`, sin aislamiento transaccional explícito |
| **Despliegue** | Docker + GitHub Actions (deploy directo, sin gate de CI) | No se menciona Dockerfile/CI en el código compartido |
| **Documentación** | `API.md`, Postman collection, dumps de OpenAPI (aunque desactualizados) | No se menciona documentación de API aparte del código |

## 2. Qué SÍ aplica a nuestras 4 necesidades, y por qué

### 2.1 Capa de explicabilidad (XAI) — la idea más valiosa a copiar

Hoy, `AnomalyDetector.predict()` devuelve `variables_contribuyentes` como lista plana y `confianza_ml` como un solo número. Eso es mucho menos útil para el productor que lo que hace `explainability.py` del otro proyecto. Adaptado a café, propondría agregar:

- **`afinidad`**: ya lo tenemos parcialmente (`confianza_ml` = probabilidad del RandomForest), pero se puede desglosar por severidad, no solo dar el top-1.
- **`caso_limitrofe`**: si la confianza del RandomForest es baja (ej. <60%), marcar la predicción como "revisar con más cuidado" — hoy no distinguimos una predicción segura de una dudosa.
- **`factores_determinantes`**: ranking de qué variables más se alejan de lo normal para ese `tipo_proceso`, con z-score respecto a la población de ese proceso (no solo "temperatura_grano" a secas, sino "temperatura_grano: 2.3 desviaciones por encima de lo normal para lavado").
- **`lotes_similares`**: mostrarle al productor 2-3 lotes históricos con condiciones parecidas y cómo terminaron (calidad final) — hoy no existe ningún mecanismo de "referencia histórica".
- **`explicacion`**: una narrativa en lenguaje natural ("Este lote muestra temperatura de grano más alta de lo normal para lavado y humedad ambiental elevada, similar a 3 lotes anteriores que terminaron en calidad regular").

Esto aplica directamente a las 4 necesidades: mejora la calidad de la predicción de calidad (explica por qué), la de tiempo (por qué se espera más/menos tiempo), y sobre todo las recomendaciones+alertas (una alerta con explicación se atiende mejor que una alerta genérica) y la detección de anomalías (distingue anomalía "clara" de "dudosa").

### 2.2 Detección de atípicos contra el promedio de su propio proceso

El otro proyecto compara a la paciente contra el **centroide de su clúster** (si la presión sistólica supera el centroide en >15%, marca alerta). Nosotros ya tenemos algo análogo pero basado en **umbrales fijos** (Cuadro 9), no en la población real de nuestros propios lotes. Se puede complementar: calcular y guardar el promedio/desviación de cada variable **por `tipo_proceso`** (a partir de `data/processed/lecturas_limpias.csv`) y usar eso para una segunda señal de atipicidad, independiente de los umbrales fijos y del IsolationForest. Ventaja: se adapta solo si, por ejemplo, el clima de una región hace que "lo normal" cambie con el tiempo.

### 2.3 Metadata rica del modelo (`model_metadata.json`)

Hoy `scripts/train_models.py` guarda métricas en `metricas.json`, pero no guarda los centroides/estadísticos de referencia que necesitaría la idea 2.2, ni los hiperparámetros usados. Vale la pena ampliar ese archivo para incluir: hiperparámetros de cada modelo, fecha de entrenamiento, y las medias/desviaciones por `tipo_proceso` de cada feature (para alimentar tanto la detección de atípicos como la capa de explicabilidad).

### 2.4 Prácticas de productización que sí conviene adoptar

- **Dockerfile** — no vimos uno en el código compartido; el otro proyecto tiene uno simple basado en `python:slim` que sirve de plantilla directa.
- **CI/CD básico** — aunque el de ellos no corre tests antes de desplegar (lo señalan ellos mismos como debilidad), tener *algún* pipeline de GitHub Actions que sí corra `pytest` antes del deploy sería mejor que lo que tienen ellos.
- **Postman collection** y **dump de OpenAPI** — documentación práctica de bajo costo que ahorra tiempo a quien conecta la app móvil o el Servicio Gestor contra el MLL.
- **`API.md`** — documento de referencia funcional de los contratos JSON de cada endpoint (nosotros ya tenemos los schemas Pydantic, pero un documento aparte ayuda a un consumidor externo que no quiere leer código).
- **Experimentos de ablación** — el otro equipo probó qué pasa si quitan una variable (`nulliparous` vs. conteos exactos) y documentaron el impacto en el Silhouette Score. Nosotros podríamos hacer lo mismo con `delta_temp` o `tipo_proceso`: entrenar sin esa feature y comparar accuracy/f1, para justificar (o cuestionar) que valga la pena tenerla.

## 3. Qué NO aplica, y por qué

### 3.1 K-Means + KNN ("cluster-then-classify")

Este es el corazón de su proyecto, y **no aplica al nuestro**. La razón de fondo: ellos necesitaban ese patrón porque **no existían perfiles de riesgo prenatal predefinidos** — tuvieron que descubrirlos con clustering antes de poder clasificar. Nosotros ya tenemos las categorías definidas por el dominio: la calidad final (excelente/buena/regular/baja, Cuadro 8/11) y la severidad (normal/advertencia/riesgo/crítico, Cuadro 9) **ya están dadas**, no hay que "descubrirlas" con K-Means. Forzar un K-Means aquí sería sobre-ingeniería: agregaría una etapa de clustering + validación (Elbow, Silhouette, ARI) para terminar reproduciendo categorías que el documento de dominio ya nos da gratis.

Si en algún momento quisiéramos **descubrir** patrones de secado que no estén en el documento de dominio (por ejemplo, "tipos de comportamiento de secado" más allá de lavado/honey/natural), ahí sí un K-Means exploratorio tendría sentido — pero como herramienta de análisis, no como parte del pipeline de producción.

### 3.2 Módulo NLP clínico

No aplica mientras no haya texto libre en nuestro dominio (bitácoras escritas por el productor). Si en el futuro se agrega un campo de notas libres ("hoy llovió toda la tarde y el grano se ve manchado"), un mini-NLP con un catálogo cerrado de términos de café (similar a `nlp_catalog.py` pero para conceptos como "moho", "grano manchado", "olor a fermento") podría ser útil — pero es una extensión futura opcional, no parte de las 4 necesidades actuales.

### 3.3 DBSCAN / clustering jerárquico para validar K

Solo tienen sentido como parte del patrón cluster-then-classify que descartamos en 3.1. No hay "K" que elegir en nuestro pipeline actual.

### 3.4 XGBoost como alternativa

Ellos lo probaron y lo descartaron a favor de KNN por simplicidad/latencia. Nosotros ya usamos RandomForest, que es comparable en robustez a XGBoost para datasets de este tamaño (~10k filas); no hay una necesidad clara de añadir XGBoost salvo que quisiéramos exprimir un poco más de accuracy en `rf_calidad` (que, como vimos antes, tiene el desempeño más débil de los 4 modelos).

## 4. Roadmap priorizado

**Alta prioridad (impacto directo en las 4 necesidades):**
1. Capa de explicabilidad ligera (sección 2.1) — mejora recomendaciones/alertas y ambas predicciones.
2. Corregir el desajuste `data/raw/lecturas_ml_training.csv` vs. lo que espera `train_models.py` (detectado en el documento anterior) — sin esto, nada de lo demás se puede reentrenar.
3. `model_metadata.json` con centroides por `tipo_proceso` (sección 2.2 y 2.3) — habilita la detección de atípicos "en vivo" contra tus propios datos, no solo contra umbrales fijos.

**Prioridad media (calidad de ingeniería, no urgente):**
4. Ablación de features (`delta_temp`, `tipo_proceso`) para justificar el diseño actual — especialmente útil dado que `rf_calidad` tiene accuracy baja (~27%), vale la pena ver si alguna feature está estorbando.
5. Dockerfile + CI/CD con `pytest` como gate.
6. Postman collection + `API.md`.

**Baja prioridad / opcional a futuro:**
7. Clustering exploratorio (K-Means) sobre los lotes, **solo como análisis** (ej. en un notebook), para ver si hay patrones de secado no capturados por `tipo_proceso` — no para producción.
8. Mini-NLP para notas de texto libre del productor, si ese campo llega a existir.

## 5. Sketch de código: capa de explicabilidad adaptada a café

Esto es una propuesta de módulo nuevo (`app/services/explainability.py`), pensado para pegar y adaptar en tu proyecto — no reentrena nada, solo interpreta lo que ya calculan `AnomalyDetector` y `Predictor`:

```python
# app/services/explainability.py (propuesta)
from typing import Dict, List
import numpy as np
import pandas as pd

# Cargado una sola vez desde model_metadata.json (centroides por tipo_proceso),
# generado por train_models.py junto con metricas.json.
# Estructura esperada: { "lavado": {"temperatura_grano": {"media": 27.1, "std": 2.3}, ...}, ... }
CENTROIDES_POR_PROCESO: Dict[str, Dict[str, Dict[str, float]]] = {}

UMBRAL_CONFIANZA_BAJA = 60.0  # % — por debajo de esto, se marca como caso límitrofe

def z_score(valor: float, media: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (valor - media) / std

def factores_determinantes(tipo_proceso: str, features: Dict[str, float], top_n: int = 3) -> List[Dict]:
    """Rankea las variables que más se alejan de lo normal para este tipo_proceso,
    usando los centroides guardados en model_metadata.json."""
    stats = CENTROIDES_POR_PROCESO.get(tipo_proceso, {})
    filas = []
    for variable, valor in features.items():
        ref = stats.get(variable)
        if ref is None:
            continue
        z = z_score(valor, ref["media"], ref["std"])
        filas.append({"variable": variable, "valor": valor, "z_score": round(z, 2)})
    filas.sort(key=lambda f: abs(f["z_score"]), reverse=True)
    return filas[:top_n]

def caso_limitrofe(confianza_ml: float) -> bool:
    return confianza_ml < UMBRAL_CONFIANZA_BAJA

def lotes_similares(df_referencia: pd.DataFrame, tipo_proceso: str, features: Dict[str, float],
                     columnas: List[str], top_n: int = 3) -> List[Dict]:
    """Busca, en un CSV de referencia (ej. una muestra de data/processed/lecturas_limpias.csv),
    los lotes históricos mas parecidos por distancia euclidiana en las columnas numericas."""
    sub = df_referencia[df_referencia["tipo_proceso"] == tipo_proceso]
    if sub.empty:
        return []
    vector = np.array([features[c] for c in columnas])
    distancias = np.linalg.norm(sub[columnas].to_numpy() - vector, axis=1)
    idx_cercanos = np.argsort(distancias)[:top_n]
    similares = sub.iloc[idx_cercanos]
    return [
        {"id_lote": int(r["id_lote"]), "calidad_final": r.get("_calidad_final_lote"), "distancia": round(float(d), 2)}
        for r, d in zip(similares.to_dict("records"), distancias[idx_cercanos])
    ]

def explicacion_narrativa(factores: List[Dict], severidad: str) -> str:
    if not factores:
        return "El lote se encuentra dentro de los parámetros esperados."
    principales = ", ".join(f"{f['variable']} ({f['z_score']:+.1f} desviaciones)" for f in factores)
    return f"Severidad '{severidad}' explicada principalmente por: {principales}."
```

Esto se integraría en `ejecutar_pipeline()` (en `app/api/routes/inference.py`) justo después de llamar a `detector.predict()`, agregando un bloque `explicabilidad` al `InferenceResponse` con `factores_determinantes`, `caso_limitrofe`, `lotes_similares` y `explicacion`.

## 6. Qué mantener igual (fortalezas de microservicioMLL que el otro proyecto no tiene)

- **El ensamble reglas + ML** es más seguro para un dominio con consecuencias físicas reales (perder un lote de café por lluvia no detectada): las reglas garantizan que casos críticos conocidos SIEMPRE se detecten, algo que un KNN puro no garantiza si el patrón no estaba bien representado en el entrenamiento.
- **`GroupShuffleSplit` por lote** evita fuga de datos entre entrenamiento y prueba; el documento del otro equipo no menciona un control equivalente para evitar que registros de la misma paciente aparezcan en train y test.
- **Separación de severidades** (`SEVERIDADES_QUE_ALERTAN = {"riesgo", "critico"}`) evita fatiga de alertas — solo se le manda correo/alerta real al productor cuando de verdad importa, mientras que su proyecto no parece diferenciar "aviso silencioso" de "alerta accionable".
