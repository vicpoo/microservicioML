# Archivo: app/api/routes/dispositivos.py
# Carpeta: microservicioMLL/app/api/routes/
"""
Registro de tokens FCM (paso 11: despliegue). Nota de arquitectura, igual que en internal.py:
el MLL es un servicio 100% interno (no valida JWT de usuarios finales, no habla directo con la
app móvil). En un despliegue real, tu API principal / Servicio Gestor es quien recibe el token
de la app móvil (junto con el JWT del usuario, que él sí valida) y se lo reenvía al MLL con el
X-Internal-Api-Key ya puesto -- mismo patrón que /internal/lecturas/nuevas.

Endpoints protegidos con la misma X-Internal-Api-Key que el resto de rutas internas.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import verificar_api_key
from app.models.database import SessionLocal
from app.models.dispositivos_usuario import DispositivoUsuario
from app.schemas.dispositivo import DesactivarDispositivoRequest, DispositivoResponse, RegistrarDispositivoRequest

router = APIRouter(prefix="/dispositivos", tags=["dispositivos"], dependencies=[Depends(verificar_api_key)])


@router.post("/registrar", response_model=DispositivoResponse, status_code=201)
def registrar_dispositivo(request: RegistrarDispositivoRequest):
    """Upsert por (id_usuario, fcm_token): si el mismo token ya estaba registrado para ese
    usuario (p. ej. la app reenvía el token en cada arranque), se reactiva/actualiza en vez de
    duplicar la fila. Un usuario puede tener varios dispositivos activos a la vez."""
    db: Session = SessionLocal()
    try:
        existente = (
            db.query(DispositivoUsuario)
            .filter(
                DispositivoUsuario.id_usuario == request.id_usuario,
                DispositivoUsuario.fcm_token == request.fcm_token,
            )
            .first()
        )
        if existente is not None:
            existente.activo = True
            existente.plataforma = request.plataforma
            db.commit()
            db.refresh(existente)
            dispositivo = existente
        else:
            dispositivo = DispositivoUsuario(
                id_usuario=request.id_usuario,
                fcm_token=request.fcm_token,
                plataforma=request.plataforma,
                activo=True,
            )
            db.add(dispositivo)
            db.commit()
            db.refresh(dispositivo)

        return DispositivoResponse(
            id_dispositivo=dispositivo.id_dispositivo,
            id_usuario=dispositivo.id_usuario,
            plataforma=dispositivo.plataforma,
            activo=dispositivo.activo,
            fecha_registro=dispositivo.fecha_registro.isoformat() if dispositivo.fecha_registro else None,
            fecha_ultima_actualizacion=dispositivo.fecha_ultima_actualizacion.isoformat() if dispositivo.fecha_ultima_actualizacion else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


@router.post("/desactivar", status_code=200)
def desactivar_dispositivo(request: DesactivarDispositivoRequest):
    """Se llama al hacer logout / desinstalar, para dejar de mandarle push a ese dispositivo.
    No falla si el token no existía (idempotente)."""
    db: Session = SessionLocal()
    try:
        actualizado = (
            db.query(DispositivoUsuario)
            .filter(
                DispositivoUsuario.id_usuario == request.id_usuario,
                DispositivoUsuario.fcm_token == request.fcm_token,
            )
            .update({"activo": False})
        )
        db.commit()
        return {"desactivado": actualizado > 0}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


@router.get("/{id_usuario}", response_model=list[DispositivoResponse])
def listar_dispositivos(id_usuario: int):
    """Diagnóstico: ver qué dispositivos activos tiene un usuario (para depurar por qué sí/no
    le llegó una notificación)."""
    db: Session = SessionLocal()
    try:
        registros = (
            db.query(DispositivoUsuario)
            .filter(DispositivoUsuario.id_usuario == id_usuario, DispositivoUsuario.activo.is_(True))
            .all()
        )
        return [
            DispositivoResponse(
                id_dispositivo=d.id_dispositivo,
                id_usuario=d.id_usuario,
                plataforma=d.plataforma,
                activo=d.activo,
                fecha_registro=d.fecha_registro.isoformat() if d.fecha_registro else None,
                fecha_ultima_actualizacion=d.fecha_ultima_actualizacion.isoformat() if d.fecha_ultima_actualizacion else None,
            )
            for d in registros
        ]
    finally:
        db.close()
