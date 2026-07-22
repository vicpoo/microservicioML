# Definición del Problema — kajve

Sistema Inteligente de Monitoreo del Secado de Café. Proyecto Integrador,
Universidad Politécnica de Chiapas, Ingeniería en Software, 9° Cuatrimestre.

## 1. Contexto

kajve monitorea el secado de café en camas africanas (*osiles*) mediante
sensores IoT instalados en cada osil, con el objetivo de mejorar la calidad
del café final. El secado es la etapa post-cosecha que más impacta la
calidad: un secado mal controlado provoca fermentación, moho, secado
irregular y pérdida de valor comercial.

**Hardware real por osil (5 sensores, un ESP32 por cama):**

| Sensor | Variable | Nota |
|---|---|---|
| BMP280 | Temperatura ambiental, presión, altitud | **No** mide humedad relativa (ver Sección 4.1) |
| DS18B20 | Temperatura del grano | Sonda 1-Wire sumergible en la capa de café |
| BH1750 | Luz / radiación solar (lux) | |
| FC-37 | Detección de lluvia (analógico + booleano) | Sin especificación de fábrica para % de precipitación |
| Sensor capacitivo | Humedad del grano | Entrega valor crudo del ADC, requiere calibración (ver Sección 6) |

El ESP32 agrupa las 5 lecturas en un solo payload JSON y lo envía por MQTT;
el backend lo guarda en `lecturas_ambientales` (Postgres/Neon).

## 2. Objetivo general

Que el sistema, a partir de las lecturas de sensores, sea capaz de:

1. Detectar anomalías según parámetros del dominio cafetalero.
2. Generar alertas cuando una variable supera un umbral de riesgo.
3. Generar recomendaciones en lenguaje natural asociadas a cada alerta.
4. Predecir el tiempo estimado restante de secado.
5. Predecir la calidad próxima a obtener del café.
6. Predecir la probabilidad de lluvia a corto plazo.

Todo esto sin exponer nunca los datos, alertas o predicciones de un osil a
un usuario que no es su propietario (ver Sección 5).

## 3. Salidas del sistema

| Salida | Tipo de problema | Estado |
|---|---|---|
| Alertas | Motor de reglas determinista (no ML) | **Implementado y verificado** |
| Recomendaciones | Motor de plantillas parametrizadas (no ML) | **Implementado y verificado** |
| Tiempo estimado de secado | Regresión | Heurística por proceso (sin datos suficientes para entrenar aún) |
| Calidad estimada del grano | Clasificación (excelente/buena/regular/baja) | Regla basada en historial de alertas del ciclo (sin datos suficientes para entrenar aún) |
| Probabilidad de lluvia | Clasificación / nowcasting a corto plazo | Diseño: tendencia de presión atmosférica (BMP280) |

### 3.1. Alertas y recomendaciones

Motor de reglas sobre los umbrales del *Documento de Calidad del Café y
Reglas del Dominio* (Secciones 3, 4, 5 y 6), diferenciado por tipo de
proceso (lavado, honey, natural). No depende de datos históricos: puede
operar desde el primer día del piloto.

### 3.2. Tiempo estimado de secado

Regresión. Sin historial de ciclos completos todavía, arranca con una
heurística basada en los tiempos típicos por proceso (lavado 6–9 días,
honey ~8–23 días, natural 10–28 días), ajustada en tiempo real según la
tendencia de humedad del grano. Se reemplaza por un modelo entrenado
(Random Forest / Gradient Boosting) cuando existan suficientes lotes
finalizados con fecha de inicio y fin reales.

### 3.3. Calidad estimada del grano

Clasificación con 4 categorías. Arranca con el criterio del documento de
dominio (Sección 7): cero alertas de riesgo/crítico → excelente; solo
advertencias → buena; alguna alerta de riesgo → regular; alguna crítica o
riesgo no atendida → baja. Se vuelve un modelo entrenado cuando haya lotes
finalizados con calidad conocida, idealmente contrastada contra catación
real (protocolo SCA/CQI).

### 3.4. Probabilidad de lluvia

Nowcasting de corto plazo (próximos 30–60 minutos) usando la tendencia de
presión atmosférica del BMP280 (una caída sostenida anticipa lluvia). No
depende del sensor FC-37 como entrada, por lo que no se bloquea por su
calibración pendiente. Para pronóstico a más horas, se recomienda
complementar con una API meteorológica externa en vez de depender solo del
sensor local.

## 4. Decisiones de alcance tomadas

### 4.1. Sensor real: BMP280, no BME280

El hardware usa BMP280 (temperatura + presión + altitud), **no** BME280. El
BMP280 no mide humedad relativa ambiental. Por lo tanto, para la v1 del
sistema:

- **Se descartan** las reglas `humedad_ambiental_alta`, `viento_excesivo` y
  `riesgo_moho_combinado` (dependían de humedad ambiental y/o viento, y
  tampoco existe anemómetro instalado). Quedan documentadas como trabajo
  futuro si se agrega el hardware correspondiente.
- **Se mantienen** `temperatura_alta`, `lluvia_detectada`,
  `secado_estancado`, `fluctuacion_climatica` y
  `radiacion_insuficiente_prolongada`, que no dependen de esas variables.

### 4.2. Mapeo de severidad

El ENUM real en Postgres (`nivel_severidad`) es `baja/media/alta/critica`,
pero el documento de dominio usa `normal/advertencia/riesgo/crítico`.
Mapeo adoptado:

| Documento de dominio | ENUM en BD |
|---|---|
| normal | (no genera fila en `alertas`) |
| advertencia | baja |
| riesgo | media |
| crítico | critica |
| — | alta (reservado, sin uso en v1) |

### 4.3. Refinamientos surgidos de verificar con datos reales

Al correr el motor de reglas contra datos reales del piloto se detectaron
dos problemas de calidad de datos, ya corregidos en la implementación:

- **Filtro de cordura de temperatura**: aparecían lecturas físicamente
  imposibles (ej. 181.6 °C ambiental, típico de un sensor desconectado o un
  glitch de lectura). Se descartan lecturas fuera de −10 °C a 60 °C antes
  de evaluarlas como alerta.
- **Lluvia sostenida vs. blip puntual**: el sensor FC-37 mostraba el valor
  `lluvia_detectada` cambiando de true a false cada pocos segundos. El
  documento distingue "llovizna" (advertencia) de "lluvia sostenida"
  (crítico); ahora se exige que la detección se mantenga positiva de forma
  continua por al menos 3 minutos antes de escalar a crítico. Una
  detección aislada se registra solo como advertencia.

## 5. Aislamiento de alertas por usuario (enrutamiento)

Ningún usuario debe ver ni recibir alertas, recomendaciones o predicciones
de un osil que no es suyo. No hay difusión general: toda notificación se
enruta únicamente al propietario del sensor/lote que la originó.

**Cadena de enrutamiento:**

1. El ESP32 envía su payload por MQTT identificado por `mac_address` /
   `id_cola_mqtt` (tabla `sensores`). El servicio de ingesta IoT resuelve
   `mac_address → id_sensor`.
2. El servicio de ingesta resuelve `id_sensor → id_lote` activo
   (`lotes_cafe.id_sensor`) antes de insertar en `lecturas_ambientales`.
3. `lotes_cafe.id_usuario` identifica al propietario del lote. El motor de
   reglas (`pipeline.py`) ya obtiene este dato junto con cada lote
   (`obtener_lotes_en_proceso`), así que toda alerta generada ya sabe de
   quién es antes de insertarse.
4. El despacho de **push (FCM)** necesita una tabla de tokens de
   dispositivo (`dispositivos_usuario`: id_usuario, token_push,
   plataforma) — pendiente de crear. El servicio de notificaciones toma el
   `id_usuario` de la alerta y envía solo a los tokens de ese usuario.
5. El **ws-gateway** (WebSocket para la app abierta) debe mantener un mapa
   `id_usuario → conexiones activas` y emitir solo a los sockets de ese
   usuario — nunca un broadcast general.

**Nota de seguridad**: el Row Level Security que ya tiene la BD
(`lotes_cafe`, `lecturas_ambientales`) protege únicamente las consultas SQL
directas. FCM y WebSockets no pasan por RLS, así que el filtrado por
usuario debe aplicarse explícitamente en el código del servicio de
notificaciones y del gateway — no se hereda automáticamente de la base de
datos.

## 6. Restricciones y brechas de datos detectadas

- Solo existe un lote real con sensor físico (`id_lote 12`, proceso
  lavado, iniciado el 19 de julio de 2026) — sin historial de ciclos
  completos, no hay datos de entrenamiento para tiempo de secado ni
  calidad todavía.
- El sensor capacitivo de humedad de grano no está calibrado (lecturas
  fijas en el valor máximo del ADC) — bloquea la regla `secado_estancado`
  y el uso de humedad de grano en % hasta que se calibre en campo.
- El único lote marcado `finalizado` no tiene `fecha_fin_secado`
  registrada — hay que corregir esto en el backend para poder calcular
  tiempos reales de secado a futuro.
- Falta la tabla `dispositivos_usuario` para tokens de push (bloquea el
  envío real de notificaciones FCM, ver Sección 5).

## 7. Estado actual de implementación

- **Hecho y verificado** contra datos reales del piloto: motor de reglas
  de alertas (`reglas.py`), motor de recomendaciones por plantillas
  (`recomendaciones.py`), configuración de umbrales (`config.py`), acceso
  a BD con mapeo de severidad y cooldown (`db.py`), orquestador
  (`pipeline.py`), y script de verificación offline
  (`simulacion_offline.py`).
- **Pendiente de decisión/implementación**: tabla y lógica de
  enrutamiento de notificaciones por usuario (Sección 5), calibración del
  sensor de humedad de grano (`calibracion.py`), modelos entrenados de
  tiempo de secado y calidad (Secciones 3.2 y 3.3), y el modelo de
  probabilidad de lluvia (Sección 3.4).