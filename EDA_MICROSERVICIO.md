# Análisis exploratorio y preparación de datos para el microservicio

## Objetivo

Este documento explica cómo se realizó el análisis exploratorio de datos (EDA) para preparar el dataset que alimenta el microservicio de detección de anomalías.

## Pasos realizados

1. Carga del dataset
   - Se cargó el archivo de entrenamiento desde la carpeta de datos crudos.

2. Exploración inicial
   - Se revisaron las columnas, tipos de datos y distribución general del dataset.

3. Identificación de valores nulos
   - Se calculó la cantidad y el porcentaje de valores faltantes por columna.

4. Visualización
   - Se generaron histogramas para variables clave como temperatura y humedad.
   - Se construyó un mapa de calor de correlaciones para identificar relaciones entre variables.

5. Limpieza de datos
   - Se rellenaron los valores numéricos faltantes con la mediana.
   - Se eliminaron filas con datos faltantes en columnas importantes como el tipo de proceso.
   - Se guardó una versión limpia en la carpeta de datos procesados.

6. Entrenamiento preliminar del modelo
   - Se entrenó un modelo de Isolation Forest con las variables de temperatura y humedad.

7. Validación del microservicio
   - Se envió una muestra al endpoint de inferencia para comprobar que el sistema responde correctamente.

## ¿Por qué es importante esta limpieza?

La limpieza de datos mejora la calidad del entrenamiento, reduce ruido y evita que el modelo se vea afectado por valores faltantes o inconsistentes. Esto permite que el microservicio funcione de forma más estable y con resultados más confiables.

## Relación con el microservicio

El notebook muestra cómo los datos pasan desde el análisis exploratorio hasta la inferencia real en la API. Esto es importante porque el microservicio no solo debe responder correctamente, sino que también necesita datos bien preparados para que el modelo tenga mejor desempeño.
