# Funcionamiento del microservicio y su integración futura con la aplicación móvil

## 1. ¿Qué hace el microservicio?

El microservicio recibe lecturas del proceso de secado del café, valida y preprocesa los datos, ejecuta un modelo local de Machine Learning no supervisado y devuelve un resultado JSON indicando si la muestra corresponde a una anomalía.

## 2. ¿Cómo funciona internamente?

1. El cliente envía los datos a la API.
2. FastAPI recibe la petición y valida el payload.
3. El preprocesador transforma las variables en características útiles.
4. El modelo Isolation Forest evalúa si la lectura es atípica.
5. La respuesta incluye:
   - si es anomalía,
   - un score de anomalía,
   - severidad,
   - mensaje explicativo.
6. La inferencia se guarda en una base de datos local para consulta posterior.

## 3. ¿Qué aporta al proyecto?

Este componente permite detectar situaciones de riesgo como:
- humedad ambiental elevada,
- ventilación insuficiente,
- lluvia o condiciones anómalas,
- temperaturas fuera de rango.

Esto complementa el seguimiento del proceso de secado y puede servir como alerta temprana para el productor.

## 4. ¿Cómo se integrará con la aplicación móvil?

La integración futura sería la siguiente:

1. La app móvil enviará lecturas desde el dispositivo o desde sensores.
2. El microservicio responderá con una evaluación de riesgo.
3. La app mostrará una alerta al usuario si la severidad es alta.
4. La app podrá mostrar historial de alertas y recomendaciones.

## 5. Propuesta de flujo de integración

```text
Sensor / dispositivo --> API FastAPI --> Modelo ML --> Respuesta JSON --> App móvil
```

## 6. Beneficios de esta integración

- Alertas automáticas en tiempo real.
- Menor dependencia de revisión manual.
- Mayor soporte para decisiones del productor.
- Base para futuras funcionalidades como notificaciones y dashboard.

## 7. Estado actual del desarrollo

Actualmente el microservicio ya:
- expone un endpoint de inferencia,
- guarda las inferencias,
- ofrece historial,
- y documenta la API con Swagger.

Este es un primer prototipo funcional que servirá como base para la integración final con la aplicación móvil.
