# Archivo: app/core/security.py
# Carpeta: microservicioMLL/app/core/
# (pega/reemplaza este archivo en esa ruta dentro de tu proyecto)

from fastapi import Header, HTTPException

from app.core.config import get_settings


async def verificar_api_key(x_internal_api_key: str = Header(default=None)):
    """El MLL es 100% interno: no habla directo con la app móvil ni valida JWT de usuarios.
    Solo lo llaman otros servicios de tu backend (Servicio Gestor, y opcionalmente tu API
    móvil si prefiere pedirle el historial a él en vez de leer Neon directo).

    Si INTERNAL_API_KEY está vacío (desarrollo local), no se exige nada.
    En producción, ponle el mismo valor en .env de este servicio y en el llamador."""
    settings = get_settings()
    if settings.internal_api_key and x_internal_api_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Falta o es inválido el header X-Internal-Api-Key")