# Archivo: app/schemas/inference_request.py
# Carpeta: microservicioMLL/app/schemas/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from typing import Dict, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    # El llamador (Gestor, o tú mismo probando) ya sabe quién es el usuario dueño del lote.
    # Si no coincide con lotes_cafe.id_usuario, el microservicio responde 403.
    id_usuario: int = Field(description="Usuario dueño del lote")
    id_lote: Optional[int] = Field(default=None, description="Si se envía, se busca el lote en lotes_cafe para tomar su tipo_proceso y fecha_inicio_secado")
    tipo_proceso: Optional[str] = Field(default=None, description="lavado, honey o natural (se ignora si id_lote resuelve un lote existente)")
    id_sensor: Optional[int] = Field(default=None, description="Sensor de origen, si se conoce")
    timestamp: Optional[str] = None
    # Claves esperadas: temperatura_grano (DS18B20), temperatura_ambiental (BMP280),
    # humedad_grano (sensor capacitivo, crudo si no está calibrado), lluvia (FC-37,
    # 1.0/0.0 resuelto desde lluvia_detectada), luz (BH1750, lux).
    # Opcional: presion_hpa (BMP280) -- si se manda, activa la predicción de riesgo de lluvia
    # del Algoritmo Genético (ver app/services/rain_predictor.py); si no viene, esa predicción
    # simplemente queda en None, el resto del pipeline sigue igual.
    # NOTA: ya no existe humedad_ambiental (el hardware real es BMP280, no BME280;
    # ver definicion_problema_kajve.md Sección 4.1).
    lecturas: Dict[str, float]
    guardar_lectura: bool = Field(default=True, description="Si True, persiste la lectura cruda en lecturas_ambientales")